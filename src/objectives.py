"""Selected-position MLM and no-leak masked latent objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .config import ModelConfig


@dataclass(frozen=True)
class LossResult:
    loss: Tensor | None
    targets: int


class MLMHead(nn.Module):
    def __init__(self, width: int, vocab_size: int, bottleneck: int | None = None) -> None:
        super().__init__()
        self.transform = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.LayerNorm(width))
        if bottleneck is None:
            self.decoder: nn.Module = nn.Linear(width, vocab_size)
        else:
            self.decoder = nn.Sequential(
                nn.Linear(width, bottleneck),
                nn.GELU(),
                nn.Linear(bottleneck, vocab_size),
            )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.decoder(self.transform(hidden_states))


def selected_mlm_loss(logits: Tensor, input_ids: Tensor, prediction_mask: Tensor) -> LossResult:
    """Cross entropy over exactly selected entries; empty selection skips safely."""
    _check_selected(logits, input_ids, prediction_mask)
    count = int(prediction_mask.sum().item())
    if count == 0:
        return LossResult(None, 0)
    loss = functional.cross_entropy(logits[prediction_mask], input_ids[prediction_mask])
    return LossResult(loss, count)


def _check_selected(values: Tensor, input_ids: Tensor, prediction_mask: Tensor) -> None:
    if values.shape[:2] != input_ids.shape or prediction_mask.shape != input_ids.shape:
        raise ValueError("objective tensors must share [batch, length]")
    if prediction_mask.dtype != torch.bool:
        raise ValueError("prediction_mask must be boolean")


def normalized_top2_target(hidden_states: tuple[Tensor, ...], prediction_mask: Tensor) -> Tensor:
    """Mean of independently layer-normalized last two post-layer teacher states.

    Each vector x is (x - mean(x[-1])) / sqrt(var(x[-1], unbiased=False)+1e-5),
    then the last two normalized layer vectors are averaged.  Only selected vectors
    are returned and they remain detached from the teacher graph.
    """
    if len(hidden_states) < 2:
        raise ValueError("top2 target requires at least two post-layer states")
    first, second = hidden_states[-2:]
    _check_selected(first, prediction_mask, prediction_mask)
    target = (
        functional.layer_norm(first, (first.shape[-1],))
        + functional.layer_norm(second, (second.shape[-1],))
    ) / 2
    return target[prediction_mask].detach()


def final_target(hidden_state: Tensor, prediction_mask: Tensor) -> Tensor:
    _check_selected(hidden_state, prediction_mask, prediction_mask)
    return hidden_state[prediction_mask].detach()


class LinearPredictor(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.projection = nn.Linear(width, width)

    def forward(self, contextual_student: Tensor, attention_mask: Tensor) -> Tensor:
        del attention_mask
        return self.projection(contextual_student)


class AttentionPredictor(nn.Module):
    """Two-layer contextual predictor over the entire masked student sequence."""

    def __init__(self, config: ModelConfig, dropout: float) -> None:
        super().__init__()
        self.positions = nn.Embedding(config.max_length, config.width)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.width,
                    nhead=config.heads,
                    dim_feedforward=config.ffn_width,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=config.norm_first,
                )
                for _ in range(2)
            ]
        )
        self.output = nn.Linear(config.width, config.width)

    def forward(self, contextual_student: Tensor, attention_mask: Tensor) -> Tensor:
        if (
            attention_mask.dtype != torch.bool
            or attention_mask.shape != contextual_student.shape[:2]
        ):
            raise ValueError("predictor requires the full boolean valid-token mask")
        positions = torch.arange(
            contextual_student.shape[1], device=contextual_student.device
        ).unsqueeze(0)
        x = contextual_student + self.positions(positions)
        x = x.masked_fill(~attention_mask.unsqueeze(-1), 0.0)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=~attention_mask)
            x = x.masked_fill(~attention_mask.unsqueeze(-1), 0.0)
        return self.output(x)


def latent_loss(
    predictions: Tensor,
    targets: Tensor,
    prediction_mask: Tensor,
    kind: Literal["mse", "smooth_l1"],
    beta: float = 1.0,
) -> LossResult:
    _check_selected(predictions, prediction_mask, prediction_mask)
    count = int(prediction_mask.sum().item())
    if count == 0:
        return LossResult(None, 0)
    chosen = predictions[prediction_mask]
    if chosen.shape != targets.shape or targets.requires_grad:
        raise ValueError("latent targets must be detached selected vectors with matching shape")
    if kind == "mse":
        value = functional.mse_loss(chosen, targets)
    elif kind == "smooth_l1":
        value = functional.smooth_l1_loss(chosen, targets, beta=beta)
    else:
        raise ValueError(f"unknown latent loss {kind}")
    return LossResult(value, count)


def vicreg_variance_term(embeddings: Tensor, floor: float = 1.0, eps: float = 1e-4) -> Tensor:
    """Hinge on per-dimension standard deviation: punish dimensions below the floor.

    Taken from VICReg (Bardes, Ponce & LeCun, 2022). Each column of ``embeddings`` is one
    representation dimension across the samples in the batch.
    """
    if embeddings.dim() != 2:
        raise ValueError("regulariser expects a [samples, width] matrix")
    if embeddings.shape[0] < 2:
        raise ValueError("regulariser needs at least two samples")
    std = torch.sqrt(embeddings.var(dim=0, unbiased=False) + eps)
    return functional.relu(floor - std).mean()


def vicreg_covariance_term(embeddings: Tensor, eps: float = 1e-4) -> Tensor:
    """Off-diagonal covariance, squared: punish two dimensions carrying the same signal."""
    centred = embeddings - embeddings.mean(dim=0, keepdim=True)
    covariance = centred.T @ centred / max(1, embeddings.shape[0] - 1)
    off_diagonal = covariance - torch.diag_embed(torch.diagonal(covariance))
    return off_diagonal.pow(2).sum() / embeddings.shape[1]


def barlow_redundancy_term(
    embeddings: Tensor, eps: float = 1e-4, diagonal_weight: float = 0.005
) -> Tensor:
    """Barlow Twins' redundancy reduction on a single view.

    Normalise each dimension, form the correlation matrix, push the off-diagonal to zero and
    (optionally) the diagonal toward one. Barlow Twins applies this to two views; here the
    invariance half is the masked-latent loss, so only the statistics half applies.
    """
    if embeddings.dim() != 2:
        raise ValueError("regulariser expects a [samples, width] matrix")
    if embeddings.shape[0] < 2:
        raise ValueError("regulariser needs at least two samples")
    centred = embeddings - embeddings.mean(dim=0, keepdim=True)
    std = torch.sqrt(centred.pow(2).mean(dim=0) + eps)
    normalised = centred / std
    correlation = normalised.T @ normalised / max(1, embeddings.shape[0] - 1)
    on_diagonal = torch.diagonal(correlation)
    off_diagonal = correlation - torch.diag_embed(on_diagonal)
    return off_diagonal.pow(2).sum() / embeddings.shape[1] + diagonal_weight * (
        1 - on_diagonal
    ).pow(2).mean()


def collapse_regulariser(
    embeddings: Tensor,
    kind: str,
    *,
    variance_floor: float = 1.0,
    variance_weight: float = 1.0,
    covariance_weight: float = 0.04,
    diagonal_weight: float = 0.005,
    eps: float = 1e-4,
) -> Tensor | None:
    """Dispatch to the configured anti-collapse term, or None when disabled."""
    if kind == "none":
        return None
    if kind == "vicreg":
        return variance_weight * vicreg_variance_term(
            embeddings, floor=variance_floor, eps=eps
        ) + covariance_weight * vicreg_covariance_term(embeddings, eps=eps)
    if kind == "barlow":
        return barlow_redundancy_term(embeddings, eps=eps, diagonal_weight=diagonal_weight)
    raise ValueError(f"unknown regulariser {kind!r}")
