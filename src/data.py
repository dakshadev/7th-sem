"""
src/data.py

Person 2 owned module: data preparation, tokenization, masking, and loader pipeline.
Adheres strictly to the batch contract defined in docs/INTERFACES.md:
- input_ids: [B, L], int64
- masked_input_ids: [B, L], int64
- attention_mask: [B, L], bool (True = valid token, False = padding)
- prediction_mask: [B, L], bool (True = eligible selected position contributing to loss)
"""

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, PreTrainedTokenizerFast

DEFAULT_TOKENIZER_NAME = "bert-base-uncased"
DEFAULT_MAX_LENGTH = 128
DEFAULT_MASK_RATIO = 0.15
DEFAULT_MEAN_SPAN_LENGTH = 3
DEFAULT_MAX_SPAN_LENGTH = 5

# Explicit WikiText corpus labelling:
# - WikiText-2 is the small pilot corpus (~2M tokens raw, ~36k lines) for rapid prototyping and smoke tests.
# - WikiText-103 is the primary target corpus (~100M tokens raw, proposal target ~5M tokens subset).
WIKITEXT_PILOT_CONFIG = "wikitext-2-raw-v1"
WIKITEXT_TARGET_CONFIG = "wikitext-103-raw-v1"
WIKITEXT_DATASET_NAME = "Salesforce/wikitext"


def get_bert_tokenizer(tokenizer_name: str = DEFAULT_TOKENIZER_NAME) -> BertTokenizerFast:
    """
    Loads and returns the fast BERT tokenizer.
    Ensures tokenizer metadata matches project specifications.
    """
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
    return tokenizer


def get_special_token_ids(tokenizer: PreTrainedTokenizerFast) -> set:
    """
    Returns the set of all special token IDs (CLS, SEP, PAD, UNK, MASK, etc.).
    """
    return set(tokenizer.all_special_ids)


def mask_tokens_independent(
    input_ids: torch.Tensor,
    eligible_mask: torch.Tensor,
    mask_ratio: float = DEFAULT_MASK_RATIO,
    mask_token_id: int = 103,
    generator: torch.Generator | None = None,
    policy: str = "fixed_ratio",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Masks eligible positions independently.
    
    Args:
        input_ids: [L], int64 original token IDs.
        eligible_mask: [L], bool mask where True indicates an eligible token.
        mask_ratio: Target proportion of eligible tokens to mask (default: 0.15).
        mask_token_id: ID of the [MASK] token (default: 103 for BERT).
        generator: Optional PyTorch random Generator for deterministic reproducibility.
        policy: "fixed_ratio" (selects exactly round(mask_ratio * num_eligible))
                or "bernoulli" (independent Bernoulli coin flips with guarantee >= 1).

    Returns:
        masked_input_ids: [L], int64 tensor with masked positions replaced by mask_token_id.
        prediction_mask: [L], bool tensor with True at masked positions.
    """
    masked_input_ids = input_ids.clone()
    prediction_mask = torch.zeros_like(eligible_mask, dtype=torch.bool)

    eligible_indices = torch.nonzero(eligible_mask, as_tuple=False).squeeze(-1)
    num_eligible = eligible_indices.numel()

    if num_eligible == 0:
        return masked_input_ids, prediction_mask

    if policy == "fixed_ratio":
        num_to_mask = max(1, int(round(num_eligible * mask_ratio)))
        num_to_mask = min(num_to_mask, num_eligible)
        perm = torch.randperm(num_eligible, generator=generator)
        selected_eligible_idx = eligible_indices[perm[:num_to_mask]]
        prediction_mask[selected_eligible_idx] = True
    elif policy == "bernoulli":
        probs = torch.full((num_eligible,), mask_ratio, dtype=torch.float32)
        if generator is not None:
            # torch.bernoulli with generator
            sampled = torch.bernoulli(probs, generator=generator).bool()
        else:
            sampled = torch.bernoulli(probs).bool()
        
        # If no token was sampled, force at least 1 random eligible token to prevent 0-target batches
        if not sampled.any():
            fallback_idx = torch.randint(0, num_eligible, (1,), generator=generator).item()
            sampled[fallback_idx] = True
            
        selected_eligible_idx = eligible_indices[sampled]
        prediction_mask[selected_eligible_idx] = True
    else:
        raise ValueError(f"Unknown independent masking policy: {policy}")

    # Guard: prediction_mask must only be True where eligible_mask is True
    prediction_mask = prediction_mask & eligible_mask
    masked_input_ids[prediction_mask] = mask_token_id

    return masked_input_ids, prediction_mask


def mask_tokens_span(
    input_ids: torch.Tensor,
    eligible_mask: torch.Tensor,
    mask_ratio: float = DEFAULT_MASK_RATIO,
    mean_span_length: int = DEFAULT_MEAN_SPAN_LENGTH,
    max_span_length: int = DEFAULT_MAX_SPAN_LENGTH,
    mask_token_id: int = 103,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Masks eligible positions using short contiguous spans (SpanBERT / Recipe A).
    Spans are bounded by eligible tokens and never cross special tokens or padding.

    Args:
        input_ids: [L], int64 original token IDs.
        eligible_mask: [L], bool mask where True indicates an eligible token.
        mask_ratio: Target proportion of eligible tokens to mask (default: 0.15).
        mean_span_length: Geometric distribution mean span length (default: 3).
        max_span_length: Upper bound on individual span length (default: 5).
        mask_token_id: ID of the [MASK] token (default: 103).
        generator: Optional PyTorch random Generator.

    Returns:
        masked_input_ids: [L], int64 tensor with masked positions replaced by mask_token_id.
        prediction_mask: [L], bool tensor with True at masked positions.
    """
    masked_input_ids = input_ids.clone()
    prediction_mask = torch.zeros_like(eligible_mask, dtype=torch.bool)

    eligible_indices = torch.nonzero(eligible_mask, as_tuple=False).squeeze(-1)
    num_eligible = eligible_indices.numel()

    if num_eligible == 0:
        return masked_input_ids, prediction_mask

    target_masked_count = max(1, int(round(num_eligible * mask_ratio)))
    target_masked_count = min(target_masked_count, num_eligible)

    # Geometric distribution parameter p = 1 / mean_span_length
    geo_p = 1.0 / max(1.0, float(mean_span_length))

    masked_count = 0
    max_attempts = 100
    attempt = 0

    # Work with a list of currently unmasked eligible indices
    while masked_count < target_masked_count and attempt < max_attempts:
        attempt += 1
        available_mask = eligible_mask & ~prediction_mask
        available_indices = torch.nonzero(available_mask, as_tuple=False).squeeze(-1)
        if available_indices.numel() == 0:
            break

        # Pick random start from available eligible indices
        start_pos_idx = torch.randint(0, available_indices.numel(), (1,), generator=generator).item()
        start_idx = available_indices[start_pos_idx].item()

        # Sample span length from geometric distribution clamped to [1, max_span_length]
        # Geometric sample: floor(log(1-U) / log(1-p)) + 1
        u = torch.rand(1, generator=generator).item()
        u = max(1e-7, min(1.0 - 1e-7, u))
        sampled_len = int(np.floor(np.log(1.0 - u) / np.log(1.0 - geo_p))) + 1
        span_len = max(1, min(sampled_len, max_span_length, target_masked_count - masked_count))

        # Expand contiguous span forward without crossing ineligible or already-masked tokens
        for step in range(span_len):
            curr_pos = start_idx + step
            if curr_pos >= len(eligible_mask):
                break
            # Boundary check: must be eligible and not yet masked
            if not eligible_mask[curr_pos] or prediction_mask[curr_pos]:
                break
            prediction_mask[curr_pos] = True
            masked_count += 1
            if masked_count >= target_masked_count:
                break

    # If still below target and attempts exhausted, fill remaining with random eligible positions
    if masked_count < target_masked_count:
        remaining_needed = target_masked_count - masked_count
        available_mask = eligible_mask & ~prediction_mask
        avail_idx = torch.nonzero(available_mask, as_tuple=False).squeeze(-1)
        if avail_idx.numel() > 0:
            fill_count = min(remaining_needed, avail_idx.numel())
            perm = torch.randperm(avail_idx.numel(), generator=generator)
            prediction_mask[avail_idx[perm[:fill_count]]] = True

    # Guard: strictly eligible
    prediction_mask = prediction_mask & eligible_mask
    masked_input_ids[prediction_mask] = mask_token_id

    return masked_input_ids, prediction_mask


def process_single_text(
    text: str,
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = DEFAULT_MAX_LENGTH,
    mask_ratio: float = DEFAULT_MASK_RATIO,
    mask_mode: str = "independent",
    generator: torch.Generator | None = None,
    seed: int | None = None,
    mean_span_length: int = DEFAULT_MEAN_SPAN_LENGTH,
    max_span_length: int = DEFAULT_MAX_SPAN_LENGTH,
) -> dict[str, torch.Tensor]:
    """
    Tokenizes a single text string, creates eligible mask, applies masking strategy,
    and returns a dictionary meeting the batch contract (unbatched [L]).

    Contract:
    - input_ids: [L], int64
    - masked_input_ids: [L], int64
    - attention_mask: [L], bool
    - prediction_mask: [L], bool
    """
    if generator is None and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    encoding = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
        return_special_tokens_mask=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].squeeze(0)  # [L] int64
    attention_mask = encoding["attention_mask"].squeeze(0).bool()  # [L] bool
    special_tokens_mask = encoding["special_tokens_mask"].squeeze(0).bool()  # [L] bool

    # Eligible positions: valid non-padding tokens that are NOT special tokens (CLS, SEP, etc.)
    # NOTE: fast tokenizers do not flag [UNK]/[MASK] in `special_tokens_mask`, so compare
    # against the tokenizer's full special-token id list explicitly. Predicting [UNK] is
    # meaningless and violates the batch contract in docs/INTERFACES.md.
    special_ids = torch.tensor(sorted(set(int(i) for i in tokenizer.all_special_ids)), dtype=torch.long)
    reserved_mask = torch.isin(input_ids, special_ids)
    eligible_mask = attention_mask & ~special_tokens_mask & ~reserved_mask

    if mask_mode == "independent":
        masked_input_ids, prediction_mask = mask_tokens_independent(
            input_ids=input_ids,
            eligible_mask=eligible_mask,
            mask_ratio=mask_ratio,
            mask_token_id=tokenizer.mask_token_id,
            generator=generator,
        )
    elif mask_mode == "span":
        masked_input_ids, prediction_mask = mask_tokens_span(
            input_ids=input_ids,
            eligible_mask=eligible_mask,
            mask_ratio=mask_ratio,
            mean_span_length=mean_span_length,
            max_span_length=max_span_length,
            mask_token_id=tokenizer.mask_token_id,
            generator=generator,
        )
    else:
        raise ValueError(f"Unsupported mask_mode: {mask_mode}. Choose 'independent' or 'span'.")

    # Final assertion of contract invariants
    assert input_ids.dtype == torch.int64
    assert masked_input_ids.dtype == torch.int64
    assert attention_mask.dtype == torch.bool
    assert prediction_mask.dtype == torch.bool
    # prediction_mask must never touch padding or special tokens
    assert not (prediction_mask & ~eligible_mask).any(), "prediction_mask touches ineligible positions"

    return {
        "input_ids": input_ids,
        "masked_input_ids": masked_input_ids,
        "attention_mask": attention_mask,
        "prediction_mask": prediction_mask,
    }


class TextMaskingDataset(Dataset):
    """
    PyTorch Dataset providing uncorrupted and masked sequences matching docs/INTERFACES.md.
    Supports deterministic masking via seed or stochastic online masking.
    """

    def __init__(
        self,
        texts: list[str],
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = DEFAULT_MAX_LENGTH,
        mask_ratio: float = DEFAULT_MASK_RATIO,
        mask_mode: str = "independent",
        seed: int | None = None,
        mean_span_length: int = DEFAULT_MEAN_SPAN_LENGTH,
        max_span_length: int = DEFAULT_MAX_SPAN_LENGTH,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_ratio = mask_ratio
        self.mask_mode = mask_mode
        self.seed = seed
        self.mean_span_length = mean_span_length
        self.max_span_length = max_span_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.texts[idx]
        generator = None
        if self.seed is not None:
            # Deterministic per-sample generator for reproducible validation masks
            generator = torch.Generator()
            generator.manual_seed(self.seed + idx * 7919)

        return process_single_text(
            text=text,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            mask_ratio=self.mask_ratio,
            mask_mode=self.mask_mode,
            generator=generator,
            mean_span_length=self.mean_span_length,
            max_span_length=self.max_span_length,
        )


def collate_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Collates a list of sample dictionaries into batched tensors [B, L].
    Validates batch invariants:
    - input_ids: [B, L], int64
    - masked_input_ids: [B, L], int64
    - attention_mask: [B, L], bool
    - prediction_mask: [B, L], bool
    - prediction_mask is False at all padding and special tokens
    """
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
    masked_input_ids = torch.stack([item["masked_input_ids"] for item in batch], dim=0)
    attention_mask = torch.stack([item["attention_mask"] for item in batch], dim=0)
    prediction_mask = torch.stack([item["prediction_mask"] for item in batch], dim=0)

    # Invariant validation
    if (prediction_mask & ~attention_mask).any():
        raise ValueError("prediction_mask includes padding tokens")

    return {
        "input_ids": input_ids,
        "masked_input_ids": masked_input_ids,
        "attention_mask": attention_mask,
        "prediction_mask": prediction_mask,
    }


def create_dataloader(
    texts: list[str],
    tokenizer: PreTrainedTokenizerFast,
    batch_size: int = 16,
    shuffle: bool = True,
    max_length: int = DEFAULT_MAX_LENGTH,
    mask_ratio: float = DEFAULT_MASK_RATIO,
    mask_mode: str = "independent",
    seed: int | None = None,
    num_workers: int = 0,
) -> DataLoader:
    """
    Creates a DataLoader yielding batches strictly conforming to the contract.
    """
    dataset = TextMaskingDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_length=max_length,
        mask_ratio=mask_ratio,
        mask_mode=mask_mode,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_batch,
        num_workers=num_workers,
    )


def decode_masked_example(
    input_ids: torch.Tensor,
    masked_input_ids: torch.Tensor,
    prediction_mask: torch.Tensor,
    tokenizer: PreTrainedTokenizerFast,
) -> dict[str, Any]:
    """
    Inspects a single tokenized and masked sequence.
    Returns original tokens, corrupted tokens, masked token positions,
    and masked target values.
    """
    orig_tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
    masked_positions = torch.nonzero(prediction_mask, as_tuple=False).squeeze(-1).tolist()
    target_tokens = [orig_tokens[pos] for pos in masked_positions]

    return {
        "original_text": tokenizer.decode(input_ids, skip_special_tokens=True),
        "corrupted_text": tokenizer.decode(masked_input_ids, skip_special_tokens=False),
        "masked_positions": masked_positions,
        "target_tokens": target_tokens,
        "num_masked": len(masked_positions),
    }


def count_tokens_in_batch(batch: dict[str, torch.Tensor]) -> dict[str, int]:
    """
    Counts tokens according to project convention:
    - nonpadding_tokens: total valid student input tokens (sum of attention_mask)
    - masked_targets: total selected prediction positions (sum of prediction_mask)
    """
    return {
        "nonpadding_tokens": int(batch["attention_mask"].sum().item()),
        "masked_targets": int(batch["prediction_mask"].sum().item()),
    }


def load_wikitext_split(
    dataset_config: str = WIKITEXT_PILOT_CONFIG,
    split: str = "train",
    min_length_chars: int = 30,
    max_samples: int | None = None,
    dataset_name: str = WIKITEXT_DATASET_NAME,
) -> tuple[list[str], dict[str, Any]]:
    """
    Loads text segments from a specified split of WikiText.
    Explicitly distinguishes between:
      - wikitext-2-raw-v1: Small pilot corpus (~2M tokens raw).
      - wikitext-103-raw-v1: Primary target corpus (~100M tokens raw, proposal target ~5M tokens subset).

    Filters out empty lines and isolated section headers below min_length_chars.

    Returns:
        clean_texts: List of filtered document text strings.
        manifest: Dictionary with dataset metadata, revision info, and counts.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, dataset_config, split=split)
    clean_texts = []
    for item in ds:
        text = item.get("text", "").strip()
        if len(text) >= min_length_chars:
            clean_texts.append(text)
            if max_samples is not None and len(clean_texts) >= max_samples:
                break

    role_label = "pilot" if "wikitext-2" in dataset_config else "target"
    manifest = {
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "corpus_role": role_label,
        "split": split,
        "raw_examples_scanned": len(ds),
        "clean_examples_retained": len(clean_texts),
        "min_length_chars": min_length_chars,
    }
    return clean_texts, manifest


def get_dataset_loaders(
    dataset_config: str = WIKITEXT_PILOT_CONFIG,
    batch_size: int = 16,
    max_length: int = DEFAULT_MAX_LENGTH,
    mask_ratio: float = DEFAULT_MASK_RATIO,
    mask_mode: str = "independent",
    seed: int = 42,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    num_workers: int = 0,
    tokenizer: PreTrainedTokenizerFast | None = None,
) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    """
    Real dataset-loader entry point for pretraining and validation.
    Loads real WikiText training and validation splits, constructs DataLoaders
    enforcing docs/INTERFACES.md batch contracts.

    Validation loader uses deterministic per-sample seeds for reproducible evaluation masks.

    Returns:
        train_loader: DataLoader yielding batched contract tensors
        val_loader: DataLoader yielding batched contract tensors
        manifest: Dictionary containing dataset configuration, sample counts, and parameters.
    """
    if tokenizer is None:
        tokenizer = get_bert_tokenizer(DEFAULT_TOKENIZER_NAME)

    train_texts, train_manifest = load_wikitext_split(
        dataset_config=dataset_config,
        split="train",
        max_samples=max_train_samples,
    )
    val_texts, val_manifest = load_wikitext_split(
        dataset_config=dataset_config,
        split="validation",
        max_samples=max_val_samples,
    )

    train_loader = create_dataloader(
        texts=train_texts,
        tokenizer=tokenizer,
        batch_size=batch_size,
        shuffle=True,
        max_length=max_length,
        mask_ratio=mask_ratio,
        mask_mode=mask_mode,
        seed=None,
        num_workers=num_workers,
    )

    val_loader = create_dataloader(
        texts=val_texts,
        tokenizer=tokenizer,
        batch_size=batch_size,
        shuffle=False,
        max_length=max_length,
        mask_ratio=mask_ratio,
        mask_mode=mask_mode,
        seed=seed,  # Deterministic validation masks
        num_workers=num_workers,
    )

    manifest = {
        "dataset_name": WIKITEXT_DATASET_NAME,
        "dataset_config": dataset_config,
        "corpus_role": "pilot" if "wikitext-2" in dataset_config else "target",
        "train_samples": len(train_texts),
        "val_samples": len(val_texts),
        "batch_size": batch_size,
        "max_length": max_length,
        "mask_ratio": mask_ratio,
        "mask_mode": mask_mode,
        "seed": seed,
    }

    return train_loader, val_loader, manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dataset Loader & Batch Generator CLI")
    parser.add_argument("--dataset-config", default=WIKITEXT_PILOT_CONFIG, choices=[WIKITEXT_PILOT_CONFIG, WIKITEXT_TARGET_CONFIG],
                        help="WikiText config: wikitext-2-raw-v1 (pilot) or wikitext-103-raw-v1 (target)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--max-length", type=int, default=128, help="Sequence length cutoff")
    parser.add_argument("--mask-ratio", type=float, default=0.15, help="Proportion of eligible tokens to mask")
    parser.add_argument("--mask-mode", default="independent", choices=["independent", "span"], help="Masking mode")
    parser.add_argument("--save-fixture", type=str, default=None, help="Optional path to save tiny batch tensor .pt")
    parser.add_argument("--inspect", action="store_true", help="Print inspection of first batch")
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset_config} (role: {'pilot' if 'wikitext-2' in args.dataset_config else 'target'})...")
    tokenizer = get_bert_tokenizer()
    train_loader, val_loader, manifest = get_dataset_loaders(
        dataset_config=args.dataset_config,
        batch_size=args.batch_size,
        max_length=args.max_length,
        mask_ratio=args.mask_ratio,
        mask_mode=args.mask_mode,
        max_train_samples=20,
        max_val_samples=10,
        tokenizer=tokenizer,
    )
    print(f"Loaded train loader with {len(train_loader.dataset)} samples, val loader with {len(val_loader.dataset)} samples.")
    batch = next(iter(train_loader))
    counts = count_tokens_in_batch(batch)
    print("Batch Contract Verified:")
    for k, v in batch.items():
        print(f"  {k:18s} shape={list(v.shape)} dtype={v.dtype}")
    print(f"Tokens counted: {counts}")

    if args.save_fixture:
        torch.save(batch, args.save_fixture)
        print(f"Saved fixture to {args.save_fixture}")

    if args.inspect:
        decoded = decode_masked_example(batch["input_ids"][0], batch["masked_input_ids"][0], batch["prediction_mask"][0], tokenizer)
        print(f"\nExample 1 Original:  {decoded['original_text'][:100]}...")
        print(f"Example 1 Corrupted: {decoded['corrupted_text'][:100]}...")
        print(f"Example 1 Targets ({decoded['num_masked']}): {decoded['target_tokens']}")

