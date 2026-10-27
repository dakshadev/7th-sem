"""Deterministic, diagnostic-only batches; not a corpus or evaluation pipeline."""

from __future__ import annotations

import torch


def make_synthetic_batches(
    count: int,
    batch_size: int,
    length: int,
    vocab_size: int,
    seed: int,
    pad_token_id: int = 0,
    mask_token_id: int = 3,
) -> list[dict[str, torch.Tensor]]:
    if vocab_size <= max(pad_token_id, mask_token_id, 4) or length < 4:
        raise ValueError("synthetic vocabulary/length too small")
    generator = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(count):
        ids = torch.randint(5, vocab_size, (batch_size, length), generator=generator)
        ids[:, 0], ids[:, -1] = 1, 2  # CLS/SEP-like fixed specials
        attention = torch.ones_like(ids, dtype=torch.bool)
        selected = torch.zeros_like(attention)
        selected[:, 1:-1:3] = True
        masked = ids.clone()
        masked[selected] = mask_token_id
        batches.append(
            {
                "input_ids": ids,
                "masked_input_ids": masked,
                "attention_mask": attention,
                "prediction_mask": selected,
            }
        )
    return batches


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Write a diagnostic-only trusted local tensor artifact."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=30_522)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    torch.save(
        make_synthetic_batches(
            args.count, args.batch_size, args.length, args.vocab_size, args.seed
        ),
        args.output,
    )


if __name__ == "__main__":
    main()
