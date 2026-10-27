"""Build a manifested training-batch artifact from one fixed text split.

Person 2 owns ``src/data.py``; this script only orchestrates it and refuses to
run with an implicit fallback masking implementation. The artifact format is
``torch.save(list[dict[str, Tensor]], out)`` with exactly the four documented
fields: ``input_ids``, ``masked_input_ids``, ``attention_mask`` and
``prediction_mask``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

BATCH_KEYS = ("input_ids", "masked_input_ids", "attention_mask", "prediction_mask")
BOUNDED_LIMITATION = (
    "Bounded pilot subset only: this is not the full proposed 5M-token WikiText-103 "
    "training budget and must not be reported as one."
)
DATASET_FIELD = "text"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            cwd=repo_root(),
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def import_data_module() -> Any:
    """Import Person 2's data module lazily; never vendor a competing implementation."""
    try:
        return importlib.import_module("src.data")
    except ModuleNotFoundError as error:
        missing = error.name or "unknown"
        raise RuntimeError(
            "Person 2's src/data.py is required to build contract-compliant batches. "
            "This branch does not vendor it. Check out a state that contains src/data.py "
            "(coordinate with the person2 branch) and install its dependencies. "
            f"Missing module: {missing}."
        ) from error


def cached_tokenizer_revision(tokenizer_id: str) -> str:
    """Return the observed Hugging Face cache snapshot hash when it is unambiguous."""
    try:
        from huggingface_hub import constants

        snapshots = (
            Path(constants.HF_HUB_CACHE)
            / f"models--{tokenizer_id.replace('/', '--')}"
            / "snapshots"
        )
        hashes = sorted(item.name for item in snapshots.iterdir() if item.is_dir())
    except (ImportError, OSError):
        return "not_available"
    if len(hashes) == 1:
        return hashes[0]
    if hashes:
        return "ambiguous_cache_snapshots:" + ",".join(hashes)
    return "not_available"


def load_fixture_texts(path: str | Path) -> list[str]:
    """Read a local JSON fixture; supported shapes are documented in the runbook."""
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(value, dict):
        if isinstance(value.get("texts"), list):
            value = value["texts"]
        elif isinstance(value.get("records"), list):
            value = [
                item.get("text") if isinstance(item, dict) else item for item in value["records"]
            ]
        elif isinstance(value.get("data"), list):
            value = value["data"]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(
            "offline fixture must be a JSON list of strings, or an object with a "
            "'texts', 'records', or 'data' list"
        )
    return value


def validate_split(split: str) -> None:
    if any(separator in split for separator in (",", "+")):
        raise ValueError("one split only: joining train and validation is not allowed")
    if not split.replace("-", "").replace("_", "").isalnum():
        raise ValueError(f"unsupported split name {split!r}")


def load_dataset_texts(
    dataset: str,
    config_name: str,
    split: str,
    max_records: int,
) -> tuple[list[str], dict[str, Any]]:
    validate_split(split)
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "the 'datasets' package is required for --dataset mode; install Person 2's "
            "documented dependencies or use --offline-fixture"
        ) from error
    stream = load_dataset(dataset, config_name, split=split, streaming=True)
    texts: list[str] = []
    seen_rows = 0
    scan_limit = max(max_records * 20, max_records + 100)
    for row in stream:
        seen_rows += 1
        if not isinstance(row, dict) or DATASET_FIELD not in row:
            raise ValueError(f"dataset rows must contain a {DATASET_FIELD!r} field")
        text = row[DATASET_FIELD]
        if isinstance(text, str) and text.strip():
            texts.append(text)
            if len(texts) >= max_records:
                break
        if seen_rows >= scan_limit:
            break
    return texts, {
        "kind": "datasets-streaming",
        "seen_rows": seen_rows,
        "scan_limit": scan_limit,
        "revision": "not_pinned_by_streaming_loader",
    }


def load_texts(args: argparse.Namespace) -> tuple[list[str], dict[str, Any]]:
    if args.offline_fixture:
        texts = load_fixture_texts(args.offline_fixture)
        if args.max_records is not None:
            texts = texts[: args.max_records]
        return texts, {
            "kind": "offline-fixture",
            "fixture": str(args.offline_fixture),
            "requested_max_records": args.max_records,
            "revision": "not_applicable",
        }
    validate_split(args.split)
    texts, metadata = load_dataset_texts(
        args.dataset, args.config_name, args.split, args.max_records
    )
    metadata.update(
        {
            "dataset": args.dataset,
            "config": args.config_name,
            "split": args.split,
            "requested_max_records": args.max_records,
        }
    )
    return texts, metadata


def make_batches(
    texts: list[str],
    data_module: Any,
    tokenizer: Any,
    *,
    max_length: int,
    policy: str,
    mask_ratio: float,
    seed: int,
    batch_size: int,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, int]]:
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    if max_length < 2:
        raise ValueError("max length must be at least 2")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    generator = torch.Generator().manual_seed(seed)
    records: list[dict[str, torch.Tensor]] = []
    skipped_empty_records = 0
    for text in texts:
        if not text.strip():
            skipped_empty_records += 1
            continue
        records.append(
            data_module.process_single_text(
                text=text,
                tokenizer=tokenizer,
                max_length=max_length,
                mask_ratio=mask_ratio,
                mask_mode=policy,
                generator=generator,
            )
        )
    if not records:
        raise ValueError("no nonempty records were available for tokenization")

    special_token_ids = tuple(int(item) for item in tokenizer.all_special_ids)
    vocab_size = int(tokenizer.vocab_size)
    batches: list[dict[str, torch.Tensor]] = []
    skipped_batches_without_targets = 0
    for start in range(0, len(records), batch_size):
        items = records[start : start + batch_size]
        if not any(bool(item["prediction_mask"].any()) for item in items):
            skipped_batches_without_targets += 1
            continue
        batch = data_module.collate_batch(items)
        validate_batch(batch, vocab_size=vocab_size, special_token_ids=special_token_ids)
        batches.append(batch)
    if not batches:
        raise ValueError("no batch contained an eligible masked target")
    return batches, {
        "records": len(records),
        "skipped_empty_records": skipped_empty_records,
        "skipped_batches_without_targets": skipped_batches_without_targets,
        "batches": len(batches),
        "nonpadding_tokens": int(
            sum(int(batch["attention_mask"].sum().item()) for batch in batches)
        ),
        "masked_targets": int(sum(int(batch["prediction_mask"].sum().item()) for batch in batches)),
    }


def validate_batch(
    batch: dict[str, torch.Tensor],
    *,
    vocab_size: int,
    special_token_ids: tuple[int, ...],
) -> None:
    if set(batch) != set(BATCH_KEYS):
        raise ValueError(f"batch keys must be exactly {sorted(BATCH_KEYS)}")
    reference = batch["input_ids"]
    if reference.dtype != torch.int64 or batch["masked_input_ids"].dtype != torch.int64:
        raise ValueError("input_ids and masked_input_ids must be int64")
    if batch["attention_mask"].dtype != torch.bool:
        raise ValueError("attention_mask must be boolean")
    if batch["prediction_mask"].dtype != torch.bool:
        raise ValueError("prediction_mask must be boolean")
    if reference.ndim != 2 or reference.shape[0] == 0 or reference.shape[1] == 0:
        raise ValueError("batch must have nonempty [batch, length] tensors")
    if any(value.shape != reference.shape for value in batch.values()):
        raise ValueError("all batch fields must share [batch, length]")
    if not batch["attention_mask"].any(dim=1).all():
        raise ValueError("each example must have at least one valid token")
    for name in ("input_ids", "masked_input_ids"):
        values = batch[name]
        if values.min().item() < 0 or values.max().item() >= vocab_size:
            raise ValueError(f"{name} contains IDs outside the tokenizer vocabulary")
    selected = batch["prediction_mask"]
    if (selected & ~batch["attention_mask"]).any():
        raise ValueError("prediction_mask touches padding")
    specials = torch.zeros_like(selected)
    for token_id in special_token_ids:
        specials |= batch["input_ids"].eq(token_id)
    if (selected & specials).any():
        raise ValueError("prediction_mask touches a special-token position")
    if (batch["masked_input_ids"][~selected] != reference[~selected]).any():
        raise ValueError("unselected masked_input_ids must equal input_ids")
    if not selected.any():
        raise ValueError("batch has no masked target")


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json_atomic(path: str | Path, value: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(destination)


def manifest_path_for(artifact: str | Path) -> Path:
    path = Path(artifact)
    return path.with_name(path.stem + ".manifest.json")


def build_batches_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the builder and return the written manifest."""
    if not args.offline_fixture:
        validate_split(args.split)
    if args.max_records is not None and args.max_records < 1:
        raise ValueError("--max-records must be positive")
    if args.mask_ratio <= 0 or args.mask_ratio >= 1:
        raise ValueError("--mask-ratio must be in (0, 1)")
    if args.policy not in {"independent", "span"}:
        raise ValueError("--policy must be 'independent' or 'span'")

    data_module = import_data_module()
    tokenizer = data_module.get_bert_tokenizer(args.tokenizer)
    texts, source = load_texts(args)
    if not texts:
        raise ValueError("no nonempty texts were selected from the requested source")
    batches, counts = make_batches(
        texts,
        data_module,
        tokenizer,
        max_length=args.max_length,
        policy=args.policy,
        mask_ratio=args.mask_ratio,
        seed=args.seed,
        batch_size=args.batch_size,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    torch.save(batches, temporary)
    temporary.replace(out)

    text_hash = hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode("utf-8")).hexdigest()
    source.update(
        {
            "dataset": args.dataset,
            "config": args.config_name,
            "split": args.split,
            "records_used": counts["records"],
            "skipped_empty_records": counts["skipped_empty_records"],
            "source_texts_sha256": text_hash,
        }
    )
    source_limitations = (
        [
            "Dataset streaming order is not revision-pinned; source_texts_sha256 "
            "identifies the exact selected subset."
        ]
        if source["kind"] == "datasets-streaming"
        else ["Offline fixture texts are local test input, not a corpus release."]
    )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "producer": "scripts/build_batches.py",
        "code_commit": git_commit(),
        "artifact": {
            "path": str(out),
            "sha256": sha256_file(out),
            "bytes": out.stat().st_size,
            "format": "torch.save(list[dict[str, torch.Tensor]])",
            "keys": list(BATCH_KEYS),
            "dtypes": {
                "input_ids": "int64",
                "masked_input_ids": "int64",
                "attention_mask": "bool",
                "prediction_mask": "bool",
            },
        },
        "source": source,
        "tokenizer": {
            "id": args.tokenizer,
            "revision": cached_tokenizer_revision(args.tokenizer),
            "vocab_size": int(tokenizer.vocab_size),
            "mask_token_id": int(tokenizer.mask_token_id),
            "special_token_ids": [int(item) for item in tokenizer.all_special_ids],
        },
        "masking": {
            "policy": args.policy,
            "ratio": args.mask_ratio,
            "seed": args.seed,
            "sampler": "src.data.process_single_text",
            "corruption": "replace selected positions with tokenizer.mask_token_id",
        },
        "counts": counts,
        "observed_masked_targets_per_input_token": (
            counts["masked_targets"] / counts["nonpadding_tokens"]
            if counts["nonpadding_tokens"]
            else 0.0
        ),
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "limitations": [
            BOUNDED_LIMITATION,
            "Exactly one split is read; train and validation are never joined.",
            *source_limitations,
            "This artifact is machine-local and is not a durable collaborator handle.",
            "No downstream geometry, probe, or transfer claim is established by this file.",
        ],
    }
    destination = manifest_path_for(out)
    write_json_atomic(destination, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a manifested, contract-valid batch artifact from one split."
    )
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--config-name", default="wikitext-103-raw-v1")
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer", default="bert-base-uncased")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--policy", choices=("independent", "span"), default="independent")
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    parser.add_argument("--max-records", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--offline-fixture",
        help="read texts from a local JSON file instead of the dataset source",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = build_batches_from_args(args)
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        parser.exit(2, f"build_batches: error: {error}\n")
    print(
        json.dumps(
            {
                "artifact": manifest["artifact"]["path"],
                "sha256": manifest["artifact"]["sha256"],
                "manifest": str(manifest_path_for(args.out)),
                "counts": manifest["counts"],
            },
            indent=2,
        )
    )
    return manifest


if __name__ == "__main__":
    main()
