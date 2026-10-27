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
    def __init__(self, width: int, vocab_size: int) -> None:
        super().__init__()
        self.transform = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.LayerNorm(width))
        self.decoder = nn.Linear(width, vocab_size)

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
