"""Tests for in-loop mask resampling, shuffling and 80/10/10 corruption."""

from __future__ import annotations

import torch

from scripts import build_batches
from src.masking import MaskResampler


def make_batch() -> dict[str, torch.Tensor]:
    input_ids = torch.tensor(
        [
            [101, 5, 6, 7, 8, 102, 0, 0],
            [101, 9, 10, 11, 12, 102, 0, 0],
        ],
        dtype=torch.long,
    )
    attention = torch.tensor(
        [
            [True, True, True, True, True, True, False, False],
            [True, True, True, True, True, True, False, False],
        ]
    )
    masked = input_ids.clone()
    masked[:, 2] = 103
    selected = torch.zeros_like(input_ids, dtype=torch.bool)
    selected[:, 2] = True
    return {
        "input_ids": input_ids,
        "masked_input_ids": masked,
        "attention_mask": attention,
        "prediction_mask": selected,
    }


def resampler(seed: int = 3, policy: str = "independent") -> MaskResampler:
    return MaskResampler(
        policy=policy,
        ratio=0.5,
        special_token_ids=(0, 100, 101, 102, 103),
        mask_token_id=103,
        mean_span_length=3,
        max_span_length=5,
        seed=seed,
    )


def test_resampler_preserves_contract_and_draws_new_masks() -> None:
    batch = make_batch()
    first = resampler(seed=3)(batch)
    second = resampler(seed=4)(batch)

    for output in (first, second):
        assert set(output) == set(batch)
        selected = output["prediction_mask"]
        assert not (selected & ~output["attention_mask"]).any()
        assert not output["input_ids"][selected].eq(101).any()
        unselected = ~selected
        torch.testing.assert_close(
            output["masked_input_ids"][unselected], output["input_ids"][unselected]
        )
        assert selected.any()
    assert not torch.equal(first["prediction_mask"], second["prediction_mask"])


def test_resampler_is_deterministic_for_a_seed() -> None:
    batch = make_batch()
    left = resampler(seed=11)(batch)
    right = resampler(seed=11)(batch)
    torch.testing.assert_close(left["prediction_mask"], right["prediction_mask"])
    other = resampler(seed=12)(batch)
    assert not torch.equal(left["prediction_mask"], other["prediction_mask"])


def test_resampler_supports_span_policy() -> None:
    batch = make_batch()
    output = resampler(seed=4, policy="span")(batch)
    assert output["prediction_mask"].any()
    assert not output["prediction_mask"][:, -2:].any()


def test_bert_corruption_mixes_replacements_and_keeps_the_mask() -> None:
    from src.data import mask_tokens_independent

    input_ids = torch.arange(10, 60, dtype=torch.long)
    eligible = torch.ones_like(input_ids, dtype=torch.bool)
    plain_masked, plain_selected = mask_tokens_independent(
        input_ids, eligible, mask_ratio=0.5, generator=torch.Generator().manual_seed(5)
    )
    corrupted, corrupted_selected = mask_tokens_independent(
        input_ids,
        eligible,
        mask_ratio=0.5,
        generator=torch.Generator().manual_seed(5),
        corruption="bert_80_10_10",
        vocab_size=64,
    )
    torch.testing.assert_close(plain_selected, corrupted_selected)
    assert plain_masked[plain_selected].eq(103).all()
    corrupted_tokens = corrupted[corrupted_selected]
    assert not corrupted_tokens.eq(103).all()
    assert corrupted_tokens.ne(input_ids[corrupted_selected]).any()
    torch.testing.assert_close(
        corrupted[~corrupted_selected], input_ids[~corrupted_selected]
    )


def test_bert_corruption_requires_vocab_size() -> None:
    from src.data import mask_tokens_span

    input_ids = torch.arange(10, 30, dtype=torch.long)
    eligible = torch.ones_like(input_ids, dtype=torch.bool)
    try:
        mask_tokens_span(
            input_ids,
            eligible,
            mask_ratio=0.5,
            generator=torch.Generator().manual_seed(1),
            corruption="bert_80_10_10",
        )
    except ValueError as error:
        assert "vocab_size" in str(error)
    else:
        raise AssertionError("expected a ValueError for a missing vocab size")


class RecordingTokenizer:
    # The recording module encodes the record index as the token id (0-5), so only an
    # unused id may be declared special or validation will reject the batch.
    all_special_ids = [100]
    vocab_size = 8


class RecordingDataModule:
    def process_single_text(self, **kwargs) -> dict[str, torch.Tensor]:
        index = int(kwargs["text"])
        ids = torch.full((4,), index, dtype=torch.long)
        masked = ids.clone()
        selected = torch.zeros(4, dtype=torch.bool)
        selected[0] = True
        return {
            "input_ids": ids,
            "masked_input_ids": masked,
            "attention_mask": torch.ones(4, dtype=torch.bool),
            "prediction_mask": selected,
        }

    def collate_batch(self, items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {key: torch.stack([item[key] for item in items]) for key in items[0]}


def run_order(shuffle: bool) -> list[int]:
    texts = [str(index) for index in range(6)]
    batches, _ = build_batches.make_batches(
        texts,
        RecordingDataModule(),
        tokenizer=RecordingTokenizer(),
        max_length=4,
        policy="independent",
        mask_ratio=0.5,
        seed=5,
        batch_size=2,
        shuffle=shuffle,
    )
    return [int(batch["input_ids"][row, 0]) for batch in batches for row in range(2)]


def test_shuffle_changes_batch_order_deterministically() -> None:
    unshuffled = run_order(False)
    assert unshuffled == [0, 1, 2, 3, 4, 5]
    shuffled = run_order(True)
    assert sorted(shuffled) == list(range(6))
    assert shuffled != unshuffled
    assert run_order(True) == shuffled
