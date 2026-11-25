"""Resample the batch contract's mask fields inside the training loop.

A batch artifact stores one mask realisation. Trusting it means every epoch and every seed
sees the same masks, so two "seeds" bound initialisation variance only. ``MaskResampler``
rebuilds ``masked_input_ids`` and ``prediction_mask`` from ``input_ids`` with a seeded
generator, so each batch gets a fresh draw and each run gets an independent stream.

The runner stays adapter-neutral: it accepts any ``Callable[[Batch], Batch]``. This class is
the concrete adapter over Person 2's masking primitives in ``src.data``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from .config import ExperimentConfig
from .data import mask_tokens_independent, mask_tokens_span

BATCH_KEYS = {"input_ids", "masked_input_ids", "attention_mask", "prediction_mask"}


@dataclass
class MaskResampler:
    policy: str
    ratio: float
    special_token_ids: tuple[int, ...]
    mask_token_id: int
    mean_span_length: int
    max_span_length: int
    corruption: str = "mask"
    vocab_size: int = 30_522
    seed: int = 0
    generator: torch.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.policy not in {"independent", "span"}:
            raise ValueError("policy must be 'independent' or 'span'")
        if not self.special_token_ids:
            raise ValueError("special_token_ids must not be empty")
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> MaskResampler:
        return cls(
            policy=config.mask.policy,
            ratio=config.mask.ratio,
            special_token_ids=tuple(config.model.special_token_ids),
            mask_token_id=config.mask.mask_token_id,
            mean_span_length=config.mask.span_length,
            max_span_length=config.mask.max_span_length,
            corruption=config.mask.corruption,
            vocab_size=config.model.vocab_size,
            seed=config.run.seed,
        )

    def __call__(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if set(batch) != BATCH_KEYS:
            raise ValueError(f"batch keys must be exactly {sorted(BATCH_KEYS)}")
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        if input_ids.dtype != torch.long:
            raise ValueError("input_ids must be int64")
        if attention_mask.dtype != torch.bool or attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must be a matching boolean tensor")
        specials = torch.tensor(self.special_token_ids, dtype=input_ids.dtype)
        eligible = attention_mask & ~torch.isin(input_ids, specials)

        masked_rows: list[Tensor] = []
        selected_rows: list[Tensor] = []
        for row_ids, row_eligible in zip(input_ids, eligible, strict=True):
            if self.policy == "span":
                masked, selected = mask_tokens_span(
                    row_ids,
                    row_eligible,
                    mask_ratio=self.ratio,
                    mean_span_length=self.mean_span_length,
                    max_span_length=self.max_span_length,
                    mask_token_id=self.mask_token_id,
                    generator=self.generator,
                    corruption=self.corruption,
                    vocab_size=self.vocab_size,
                )
            else:
                masked, selected = mask_tokens_independent(
                    row_ids,
                    row_eligible,
                    mask_ratio=self.ratio,
                    mask_token_id=self.mask_token_id,
                    generator=self.generator,
                    corruption=self.corruption,
                    vocab_size=self.vocab_size,
                )
            masked_rows.append(masked)
            selected_rows.append(selected)

        return {
            **batch,
            "masked_input_ids": torch.stack(masked_rows, dim=0),
            "prediction_mask": torch.stack(selected_rows, dim=0),
        }
