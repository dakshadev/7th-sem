"""A small random-initialized, padding-aware Transformer encoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import ModelConfig


@dataclass(frozen=True)
class EncoderOutput:
    last_hidden_state: Tensor
    hidden_states: tuple[Tensor, ...] = ()


class SmallEncoder(nn.Module):
    """Learned token/absolute-position encoder; True attention entries are valid."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(
            config.vocab_size, config.width, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(config.max_length, config.width)
        self.embedding_norm = nn.LayerNorm(config.width)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.width,
                    nhead=config.heads,
                    dim_feedforward=config.ffn_width,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=config.norm_first,
                )
                for _ in range(config.layers)
            ]
        )

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor, return_hidden_states: bool = False
    ) -> EncoderOutput:
        if input_ids.dtype != torch.long:
            raise ValueError("input_ids must be int64")
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must be matching [batch, length]")
        if input_ids.shape[0] == 0:
            raise ValueError("batch must contain at least one example")
        if attention_mask.dtype != torch.bool:
            raise ValueError("attention_mask must be boolean with True for valid tokens")
        if input_ids.shape[1] > self.config.max_length or input_ids.shape[1] == 0:
            raise ValueError("sequence length must be between 1 and configured max_length")
        if input_ids.min().item() < 0 or input_ids.max().item() >= self.config.vocab_size:
            raise ValueError("input_ids outside configured vocabulary")
        if not attention_mask.any(dim=1).all():
            raise ValueError("each example must contain at least one valid token")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        x = self.token_embeddings(input_ids) + self.position_embeddings(positions)
        x = self.embedding_dropout(self.embedding_norm(x))
        valid = attention_mask.unsqueeze(-1)
        x = x.masked_fill(~valid, 0.0)
        states: list[Tensor] = []
        # Transformer uses True for positions to ignore: invert shared contract here.
        padding_mask = ~attention_mask
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=padding_mask)
            x = x.masked_fill(~valid, 0.0)
            if return_hidden_states:
                states.append(x)
        return EncoderOutput(x, tuple(states))
