"""
tests/test_data.py

Unit tests for Person 2 data pipeline (src/data.py).
Deterministic CPU unit tests with local fixtures and no network calls.
"""

import pytest
import torch

from src.data import (
    TextMaskingDataset,
    collate_batch,
    count_tokens_in_batch,
    create_dataloader,
    decode_masked_example,
    get_bert_tokenizer,
    process_single_text,
)


@pytest.fixture(scope="module")
def tokenizer():
    """Loads tokenizer from local cache."""
    return get_bert_tokenizer("bert-base-uncased")


@pytest.fixture
def sample_texts():
    return [
        "Language models learn useful representations of text through self-supervised objectives.",
        "The masked latent prediction framework compares teacher targets against masked language modeling.",
        "Undergraduate research requires clear experimental protocols, deterministic unit tests, and rigorous evaluation.",
        "Short sentence for padding test.",
    ]


def test_tokenizer_metadata(tokenizer):
    """Verify tokenizer vocab size and special token IDs match project context."""
    assert tokenizer.vocab_size == 30522
    assert tokenizer.pad_token_id == 0
    assert tokenizer.cls_token_id == 101
    assert tokenizer.sep_token_id == 102
    assert tokenizer.mask_token_id == 103


def test_batch_contract_shapes_and_dtypes(tokenizer, sample_texts):
    """Verify shapes, dtypes, and ranges according to docs/INTERFACES.md."""
    loader = create_dataloader(sample_texts, tokenizer, batch_size=4, shuffle=False, max_length=128, seed=42)
    batch = next(iter(loader))

    assert "input_ids" in batch
    assert "masked_input_ids" in batch
    assert "attention_mask" in batch
    assert "prediction_mask" in batch

    # Shapes: [B, L] with L=128
    assert batch["input_ids"].shape == (4, 128)
    assert batch["masked_input_ids"].shape == (4, 128)
    assert batch["attention_mask"].shape == (4, 128)
    assert batch["prediction_mask"].shape == (4, 128)

    # Dtypes
    assert batch["input_ids"].dtype == torch.int64
    assert batch["masked_input_ids"].dtype == torch.int64
    assert batch["attention_mask"].dtype == torch.bool
    assert batch["prediction_mask"].dtype == torch.bool

    # Token ID ranges
    assert (batch["input_ids"] >= 0).all()
    assert (batch["input_ids"] < tokenizer.vocab_size).all()
    assert (batch["masked_input_ids"] >= 0).all()
    assert (batch["masked_input_ids"] < tokenizer.vocab_size).all()


def test_special_tokens_and_padding_exclusion(tokenizer, sample_texts):
    """
    prediction_mask must be False at padding and all tokenizer special tokens
    (CLS, SEP, PAD, etc.).
    """
    special_ids = set(tokenizer.all_special_ids)
    dataset = TextMaskingDataset(sample_texts, tokenizer, max_length=128, mask_ratio=0.15, seed=123)

    for i in range(len(dataset)):
        item = dataset[i]
        input_ids = item["input_ids"]
        pred_mask = item["prediction_mask"]
        att_mask = item["attention_mask"]

        # 1. No padding position can be in prediction_mask
        padding_positions = ~att_mask
        assert not (pred_mask & padding_positions).any(), f"Padding masked at item {i}"

        # 2. No special token can be in prediction_mask
        for pos in torch.nonzero(pred_mask, as_tuple=False).squeeze(-1):
            token_id = input_ids[pos].item()
            assert token_id not in special_ids, f"Special token {token_id} at pos {pos} was masked!"

        # Position 0 is always [CLS]
        assert not pred_mask[0].item(), "CLS was masked!"


def test_corruption_preserves_unmasked_tokens(tokenizer, sample_texts):
    """
    Unmasked positions must remain completely unaltered.
    Masked positions must equal mask_token_id.
    """
    dataset = TextMaskingDataset(sample_texts, tokenizer, max_length=128, mask_ratio=0.15, seed=99)
    for i in range(len(dataset)):
        item = dataset[i]
        input_ids = item["input_ids"]
        masked_ids = item["masked_input_ids"]
        pred_mask = item["prediction_mask"]

        # Unselected positions unchanged
        assert torch.equal(input_ids[~pred_mask], masked_ids[~pred_mask])

        # Selected positions replaced by [MASK] (103)
        if pred_mask.any():
            assert (masked_ids[pred_mask] == tokenizer.mask_token_id).all()


def test_reproducibility_by_seed(tokenizer, sample_texts):
    """Deterministic seed produces identical masks; different seeds produce different masks."""
    ds1 = TextMaskingDataset(sample_texts, tokenizer, max_length=128, mask_ratio=0.15, seed=42)
    ds2 = TextMaskingDataset(sample_texts, tokenizer, max_length=128, mask_ratio=0.15, seed=42)
    ds3 = TextMaskingDataset(sample_texts, tokenizer, max_length=128, mask_ratio=0.15, seed=999)

    item1 = ds1[0]
    item2 = ds2[0]
    item3 = ds3[0]

    assert torch.equal(item1["prediction_mask"], item2["prediction_mask"])
    assert torch.equal(item1["masked_input_ids"], item2["masked_input_ids"])

    # Different seeds should produce different mask patterns
    assert not torch.equal(item1["prediction_mask"], item3["prediction_mask"])


def test_span_masking_behavior(tokenizer, sample_texts):
    """Span masking creates valid contiguous chunks bounded by eligible tokens."""
    dataset = TextMaskingDataset(
        sample_texts,
        tokenizer,
        max_length=128,
        mask_ratio=0.15,
        mask_mode="span",
        seed=101,
        mean_span_length=3,
        max_span_length=5,
    )

    special_ids = set(tokenizer.all_special_ids)
    for i in range(len(dataset)):
        item = dataset[i]
        pred_mask = item["prediction_mask"]
        att_mask = item["attention_mask"]
        input_ids = item["input_ids"]

        # Invariants: no padding, no special tokens
        assert not (pred_mask & ~att_mask).any()
        for pos in torch.nonzero(pred_mask, as_tuple=False).squeeze(-1):
            assert input_ids[pos].item() not in special_ids


def test_no_target_and_empty_text_handling(tokenizer):
    """Empty string or pure whitespace must not crash or produce invalid masks."""
    empty_item = process_single_text("", tokenizer, max_length=128)
    assert empty_item["input_ids"].shape == (128,)
    assert empty_item["prediction_mask"].sum().item() == 0
    assert not (empty_item["prediction_mask"] & ~empty_item["attention_mask"]).any()

    # Collation with empty text included
    dataset = TextMaskingDataset(["", "Valid short text."], tokenizer, max_length=128, seed=42)
    batch = collate_batch([dataset[0], dataset[1]])
    assert batch["input_ids"].shape == (2, 128)
    assert batch["prediction_mask"][0].sum().item() == 0
    assert batch["prediction_mask"][1].sum().item() > 0


def test_decode_masked_example(tokenizer):
    """Verify inspection and decoding utility."""
    text = "Machine learning models need rigorous empirical evaluation."
    item = process_single_text(text, tokenizer, max_length=32, mask_ratio=0.25, seed=12)
    decoded = decode_masked_example(
        item["input_ids"], item["masked_input_ids"], item["prediction_mask"], tokenizer
    )
    assert "original_text" in decoded
    assert "corrupted_text" in decoded
    assert "[MASK]" in decoded["corrupted_text"]
    assert decoded["num_masked"] == len(decoded["target_tokens"])


def test_count_tokens_in_batch(tokenizer, sample_texts):
    """Verify nonpadding tokens and masked targets counting convention."""
    loader = create_dataloader(sample_texts, tokenizer, batch_size=4, shuffle=False, max_length=128, seed=42)
    batch = next(iter(loader))
    counts = count_tokens_in_batch(batch)
    assert counts["nonpadding_tokens"] > 0
    assert counts["masked_targets"] > 0
    assert counts["masked_targets"] < counts["nonpadding_tokens"]


def test_get_dataset_loaders_offline(tokenizer, monkeypatch):
    """Verify get_dataset_loaders returns valid DataLoaders and manifest offline without network."""
    from src import data

    dummy_train = [
        "First research sample text for unit testing the real loader entry point.",
        "Second text sample ensuring sufficient sequence length and token variety.",
    ]
    dummy_val = [
        "Held out validation sequence for deterministic evaluation masking testing.",
    ]

    def mock_load_wikitext_split(dataset_config, split, min_length_chars=30, max_samples=None, **kwargs):
        texts = dummy_train if split == "train" else dummy_val
        manifest = {
            "dataset_config": dataset_config,
            "corpus_role": "pilot" if "wikitext-2" in dataset_config else "target",
            "split": split,
            "clean_examples_retained": len(texts),
        }
        return texts, manifest

    monkeypatch.setattr(data, "load_wikitext_split", mock_load_wikitext_split)

    train_loader, val_loader, manifest = data.get_dataset_loaders(
        dataset_config=data.WIKITEXT_PILOT_CONFIG,
        batch_size=2,
        max_length=64,
        tokenizer=tokenizer,
    )

    assert manifest["corpus_role"] == "pilot"
    assert manifest["dataset_config"] == data.WIKITEXT_PILOT_CONFIG
    assert manifest["train_samples"] == 2
    assert manifest["val_samples"] == 1

    train_batch = next(iter(train_loader))
    assert train_batch["input_ids"].shape == (2, 64)
    assert train_batch["prediction_mask"].shape == (2, 64)

    val_batch = next(iter(val_loader))
    assert val_batch["input_ids"].shape == (1, 64)



def test_unk_and_mask_positions_are_never_selected() -> None:
    """Fast tokenizers leave [UNK]/[MASK] out of special_tokens_mask, so masking must
    exclude the tokenizer's full special-id list. Predicting [UNK] is meaningless and
    breaks the docs/INTERFACES.md batch contract."""
    from src.data import get_bert_tokenizer, process_single_text

    tokenizer = get_bert_tokenizer()
    reserved = {int(i) for i in tokenizer.all_special_ids}
    # Characters outside the uncased vocabulary force [UNK] tokens into the sequence.
    texts = [
        "the quick brown fox \u0d2a\u0d4d\u0d30\u0d1c\u0d15\u0d4d\u0d1f\u0d3f jumps over 42 lazy dogs",
        "researchers recorded \u0416\u0438\u0437\u043d\u044c scores and \u4e2d\u6587 labels in the corpus",
        "a" * 20 + " \u0411\u0411\u0411 " + "b" * 20,
    ]
    for text in texts:
        for mode in ("independent", "span"):
            out = process_single_text(
                text, tokenizer, max_length=64, mask_ratio=0.5, mask_mode=mode, seed=7
            )
            ids, selected = out["input_ids"], out["prediction_mask"]
            assert not any(
                bool(selected[i]) and int(ids[i]) in reserved for i in range(len(ids))
            ), f"{mode} selected a reserved token position"
            assert bool(selected.any()), f"{mode} selected nothing in a long enough text"
