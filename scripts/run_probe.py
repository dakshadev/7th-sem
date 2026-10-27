"""Frozen linear probe (AG News) for Person 1 student exports.

Contract (docs/INTERFACES.md, docs/EXPERIMENT_PROTOCOL.md):
- The student encoder is frozen and in eval mode, receives UNCORRUPTED text.
- Sentence vectors come from Person 2's attention-mask-aware mean pooling that excludes
  padding and every special token.
- The scaler and classifier are fit on probe-train data only; the held-out split is used once.
- Every reported number comes from this execution; nothing is imputed.

This measures transfer of a *bounded pilot* student. It is not evidence that one design
choice beats another at 50 updates on ~23k tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from src.train import load_exported_student  # noqa: F401  (re-exported for callers)

PROBE_DATASET = "fancyzhx/ag_news"
AG_NEWS_LABELS = ("World", "Sports", "Business", "Sci/Tech")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_probe_split(
    split: str, limit: int, dataset: str = PROBE_DATASET
) -> tuple[list[str], list[int], dict[str, Any]]:
    from datasets import load_dataset

    data = load_dataset(dataset, split=f"{split}[:{limit}]")
    texts = [str(row["text"]) for row in data]
    labels = [int(row["label"]) for row in data]
    meta = {
        "dataset": dataset,
        "split": split,
        "requested": limit,
        "used": len(texts),
        "class_counts": {
            int(k): int(v)
            for k, v in zip(*np.unique(labels, return_counts=True), strict=True)
        },
    }
    return texts, labels, meta


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    max_length: int,
    batch_size: int,
    device: torch.device,
    special_token_ids: Iterable[int],
) -> np.ndarray:
    """Frozen, eval-mode, uncorrupted features via Person 2's masked mean pooling."""
    from src.evaluate import masked_mean_pooling

    model.eval()
    special = set(int(i) for i in special_token_ids)
    vectors: list[Tensor] = []
    for start in range(0, len(texts), batch_size):
        chunk = list(texts[start : start + batch_size])
        encoded = tokenizer(
            chunk,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).bool()
        output = model(input_ids, attention_mask)
        hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
        pooled = masked_mean_pooling(hidden, attention_mask, input_ids, special)
        vectors.append(pooled.detach().to("cpu", dtype=torch.float32))
    return torch.cat(vectors, dim=0).numpy()


def probe_student(
    student_path: Path,
    *,
    train_limit: int,
    test_limit: int,
    tokenizer_name: str,
    max_length: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    from src.data import get_bert_tokenizer
    from src.evaluate import evaluate_frozen_linear_probe

    tokenizer = get_bert_tokenizer(tokenizer_name)
    special_ids = tokenizer.all_special_ids

    train_texts, train_labels, train_meta = _load_probe_split("train", train_limit)
    test_texts, test_labels, test_meta = _load_probe_split("test", test_limit)

    model = load_exported_student(student_path, device=str(device))
    model.to(device).eval()

    train_features = extract_features(
        model, tokenizer, train_texts, max_length=max_length, batch_size=batch_size,
        device=device, special_token_ids=special_ids,
    )
    test_features = extract_features(
        model, tokenizer, test_texts, max_length=max_length, batch_size=batch_size,
        device=device, special_token_ids=special_ids,
    )

    result = evaluate_frozen_linear_probe(
        train_features, np.asarray(train_labels), test_features, np.asarray(test_labels), seed=seed
    )
    return {
        "student": str(student_path),
        "student_sha256": hashlib.sha256(student_path.read_bytes()).hexdigest(),
        "train_meta": train_meta,
        "test_meta": test_meta,
        "feature_shape": list(train_features.shape),
        "pooling": (
            "masked mean over valid, non-special tokens; uncorrupted inputs; frozen eval mode"
        ),
        "labels": list(AG_NEWS_LABELS),
        **result,
    }


def random_baseline(
    *,
    train_limit: int,
    test_limit: int,
    model_config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """Fresh random-init encoder with the same architecture: the transfer floor."""
    from src.config import ExperimentConfig
    from src.data import get_bert_tokenizer
    from src.evaluate import evaluate_frozen_linear_probe
    from src.model import SmallEncoder

    torch.manual_seed(seed)
    tokenizer = get_bert_tokenizer()
    cfg = ExperimentConfig.from_dict({"model": model_config}).model
    model = SmallEncoder(cfg).to(device).eval()

    train_texts, train_labels, train_meta = _load_probe_split("train", train_limit)
    test_texts, test_labels, test_meta = _load_probe_split("test", test_limit)
    train_features = extract_features(
        model, tokenizer, train_texts, max_length=cfg.max_length, batch_size=32,
        device=device, special_token_ids=tokenizer.all_special_ids,
    )
    test_features = extract_features(
        model, tokenizer, test_texts, max_length=cfg.max_length, batch_size=32,
        device=device, special_token_ids=tokenizer.all_special_ids,
    )
    result = evaluate_frozen_linear_probe(
        train_features, np.asarray(train_labels), test_features, np.asarray(test_labels), seed=seed
    )
    return {
        "student": "random-initialised SmallEncoder (same architecture)",
        "student_sha256": None,
        "train_meta": train_meta,
        "test_meta": test_meta,
        "feature_shape": list(train_features.shape),
        **result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Frozen AG News linear probe for student exports.")
    parser.add_argument("--student", required=True, action="append", help="student.pt (repeatable)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-limit", type=int, default=4000)
    parser.add_argument("--test-limit", type=int, default=1000)
    parser.add_argument("--tokenizer", default="bert-base-uncased")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--random-baseline", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        ("cpu" if args.device == "auto" else args.device)
    )
    _set_seed(args.seed)

    report: dict[str, Any] = {
        "schema_version": 1,
        "device": str(device),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seed": args.seed,
        "probe": "frozen linear probe, StandardScaler + LogisticRegression fit on train split only",
        "students": {},
        "not_run": [
            "second seed",
            "hyperparameter search on the probe (deliberately fixed at C=1.0, max_iter=1000)",
            "larger probe splits",
        ],
    }

    for path in args.student:
        student = Path(path)
        if not student.is_file():
            raise SystemExit(f"evaluate_probe: missing student export {student}")
        print(f"probing {student} on {device} ...", flush=True)
        report["students"][student.parent.name or student.stem] = probe_student(
            student,
            train_limit=args.train_limit,
            test_limit=args.test_limit,
            tokenizer_name=args.tokenizer,
            max_length=args.max_length,
            batch_size=args.batch_size,
            seed=args.seed,
            device=device,
        )

    if args.random_baseline:
        first = next(iter(report["students"].values()))
        model_config = first.get("model_config")
        if model_config is None:
            payload = torch.load(args.student[0], map_location="cpu", weights_only=True)
            model_config = payload["model_config"]
        print("probing random-init baseline ...", flush=True)
        report["random_baseline"] = random_baseline(
            train_limit=args.train_limit,
            test_limit=args.test_limit,
            model_config=model_config,
            seed=args.seed,
            device=device,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    keys = ("accuracy", "macro_f1", "majority_baseline")
    summary = {
        name: {k: value.get(k) for k in keys} for name, value in report["students"].items()
    }
    if "random_baseline" in report:
        summary["random_baseline"] = {
            k: report["random_baseline"].get(k) for k in keys
        }
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
