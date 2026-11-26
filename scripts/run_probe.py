"""Frozen linear probe (AG News) for Person 1 student exports.

Contract (docs/INTERFACES.md, docs/EXPERIMENT_PROTOCOL.md):
- The student encoder is frozen and in eval mode, receives UNCORRUPTED text.
- Sentence vectors come from Person 2's attention-mask-aware mean pooling that excludes
  padding and every special token; [CLS] pooling is reported alongside it.
- The scaler and classifier are fit on probe-train data only; the held-out split is used once.
- Every reported number comes from this execution; nothing is imputed.

Two reference numbers accompany every student:
- ``lexical_baseline``: TF-IDF (1-2 grams) + logistic regression on the identical splits.
  AG News is largely separable by topic words, so this is the floor a "semantic" probe must
  clear before a comparison between objectives means anything.
- ``random_baseline``: a fresh random-init encoder of the same architecture, repeated over
  several seeds and reported as a mean +/- sd band rather than a single unpinned draw.

Splits are stratified and shuffled by default. The earlier head-of-split sampling produced
a 1.53x class imbalance and correlated the sample with dataset order; ``--split-strategy head``
reproduces it for comparison with the recorded runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import Tensor

if __package__ in (None, ""):
    # Running as `python scripts/run_probe.py` puts scripts/ on sys.path, so `import src`
    # fails without this. The Kaggle kernel papered over it with PYTHONPATH; this fixes it.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train import load_exported_student

PROBE_DATASET = "fancyzhx/ag_news"
AG_NEWS_LABELS = ("World", "Sports", "Business", "Sci/Tech")
Split = tuple[list[str], list[int], dict[str, Any]]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stratified_indices(labels: Sequence[int], limit: int, seed: int) -> list[int]:
    """Class-balanced shuffled indices, deterministic for a seed.

    Each class contributes floor(limit / k) examples plus one more for the first
    ``limit % k`` classes. If a class is too small, the shortfall is filled from the
    remaining pool so the result always has ``min(limit, len(labels))`` entries.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    total = min(limit, len(labels))
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        by_class.setdefault(int(label), []).append(index)
    classes = sorted(by_class)
    for label in classes:
        rng.shuffle(by_class[label])

    base, remainder = divmod(total, len(classes))
    chosen: list[int] = []
    for position, label in enumerate(classes):
        wanted = base + (1 if position < remainder else 0)
        chosen.extend(by_class[label][: min(wanted, len(by_class[label]))])

    if len(chosen) < total:
        taken = set(chosen)
        remaining = [index for index in range(len(labels)) if index not in taken]
        rng.shuffle(remaining)
        chosen.extend(remaining[: total - len(chosen)])

    rng.shuffle(chosen)
    return chosen[:total]


def summarise(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot summarise an empty sequence")
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _load_probe_split(
    split: str,
    limit: int,
    dataset: str = PROBE_DATASET,
    *,
    seed: int = 42,
    strategy: str = "stratified",
) -> Split:
    from datasets import load_dataset

    data = load_dataset(dataset, split=split)
    texts_all = [str(row["text"]) for row in data]
    labels_all = [int(row["label"]) for row in data]
    if strategy == "head":
        indices = list(range(min(limit, len(texts_all))))
    elif strategy == "stratified":
        indices = stratified_indices(labels_all, limit, seed)
    else:
        raise ValueError(f"unknown split strategy {strategy!r}")

    texts = [texts_all[index] for index in indices]
    labels = [labels_all[index] for index in indices]
    counts: dict[str, int] = {}
    for label in labels:
        counts[str(label)] = counts.get(str(label), 0) + 1
    meta = {
        "dataset": dataset,
        "split": split,
        "requested": limit,
        "used": len(texts),
        "pool_size": len(texts_all),
        "strategy": strategy,
        "sampling_seed": seed,
        "class_counts": counts,
    }
    return texts, labels, meta


def load_splits(
    *,
    train_limit: int = 4000,
    test_limit: int = 1000,
    dataset: str = PROBE_DATASET,
    seed: int = 42,
    strategy: str = "stratified",
) -> tuple[Split, Split]:
    train = _load_probe_split(
        "train", train_limit, dataset, seed=seed, strategy=strategy
    )
    test = _load_probe_split("test", test_limit, dataset, seed=seed, strategy=strategy)
    return train, test


@torch.no_grad()
def extract_feature_matrix(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    max_length: int,
    batch_size: int,
    device: torch.device,
    special_token_ids: Iterable[int],
    layer_grid: bool = True,
    input_corruption: str = "clean",
    mask_ratio: float = 0.15,
    mask_token_id: int = 103,
    corruption_seed: int = 0,
) -> dict[str, np.ndarray]:
    """Frozen, eval-mode features for every encoder layer.

    Returns ``layer_<n>_mean`` for all layers, plus ``layer_<n>_cls`` when
    ``layer_grid`` is set. Layer numbering is 1-based over post-encoder states;
    the embedding output is not a layer and is not probed.

    ``input_corruption='mask'`` feeds the encoder a deterministically masked sequence while
    the *pooling mask* is still built from the original ids. That keeps the pooled token set
    identical between clean and masked runs, so the comparison isolates the encoder's input
    rather than changing which tokens are averaged.
    """
    from src.data import mask_tokens_independent
    from src.evaluate import masked_mean_pooling

    if input_corruption not in {"clean", "mask"}:
        raise ValueError("input_corruption must be 'clean' or 'mask'")
    model.eval()
    special = set(int(item) for item in special_token_ids)
    special_tensor = torch.tensor(sorted(special), dtype=torch.long)
    generator = torch.Generator().manual_seed(corruption_seed)
    chunks: dict[str, list[Tensor]] = {}
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
        original_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).bool()
        model_ids = original_ids
        if input_corruption == "mask":
            eligible = attention_mask & ~torch.isin(original_ids, special_tensor.to(device))
            masked_rows = []
            for row_ids, row_eligible in zip(original_ids, eligible, strict=True):
                masked_row, _ = mask_tokens_independent(
                    row_ids,
                    row_eligible,
                    mask_ratio=mask_ratio,
                    mask_token_id=mask_token_id,
                    generator=generator,
                )
                masked_rows.append(masked_row)
            model_ids = torch.stack(masked_rows, dim=0)
        output = model(model_ids, attention_mask, return_hidden_states=True)
        for index, state in enumerate(output.hidden_states, start=1):
            pooled = masked_mean_pooling(state, attention_mask, original_ids, special)
            chunks.setdefault(f"layer_{index}_mean", []).append(
                pooled.detach().to("cpu", dtype=torch.float32)
            )
            if layer_grid:
                chunks.setdefault(f"layer_{index}_cls", []).append(
                    state[:, 0, :].detach().to("cpu", dtype=torch.float32)
                )
    if not chunks:
        raise ValueError("model returned no hidden states")
    return {name: torch.cat(values, dim=0).numpy() for name, values in chunks.items()}


def probe_features(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    from src.evaluate import evaluate_frozen_linear_probe

    return evaluate_frozen_linear_probe(
        train_features, train_labels, test_features, test_labels, seed=seed
    )


def probe_student(
    student_path: Path,
    *,
    train: Split,
    test: Split,
    tokenizer_name: str,
    max_length: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    layer_grid: bool = True,
    input_corruption: str = "clean",
) -> dict[str, Any]:
    from src.data import get_bert_tokenizer

    train_texts, train_labels, train_meta = train
    test_texts, test_labels, test_meta = test
    tokenizer = get_bert_tokenizer(tokenizer_name)
    model = load_exported_student(student_path, device=str(device))
    model.to(device).eval()

    features = extract_feature_matrix(
        model,
        tokenizer,
        train_texts,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
        special_token_ids=tokenizer.all_special_ids,
        layer_grid=layer_grid,
        input_corruption=input_corruption,
        corruption_seed=seed,
    )
    test_features = extract_feature_matrix(
        model,
        tokenizer,
        test_texts,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
        special_token_ids=tokenizer.all_special_ids,
        layer_grid=layer_grid,
        input_corruption=input_corruption,
        corruption_seed=seed,
    )
    by_layer = {
        name: probe_features(
            features[name],
            np.asarray(train_labels),
            test_features[name],
            np.asarray(test_labels),
            seed=seed,
        )
        for name in features
    }
    layers = len([name for name in features if name.endswith("_mean")])
    final = by_layer[f"layer_{layers}_mean"]
    return {
        "student": str(student_path),
        "student_sha256": hashlib.sha256(student_path.read_bytes()).hexdigest(),
        "train_meta": train_meta,
        "test_meta": test_meta,
        "feature_shape": list(features[f"layer_{layers}_mean"].shape),
        "pooling": (
            "masked mean over valid, non-special tokens of the original ids; frozen eval mode"
        ),
        "input_corruption": input_corruption,
        "layers": layers,
        "labels": list(AG_NEWS_LABELS),
        "accuracy": final["accuracy"],
        "macro_f1": final["macro_f1"],
        "majority_baseline": final["majority_baseline"],
        "train_samples": final["train_samples"],
        "test_samples": final["test_samples"],
        "num_classes": final["num_classes"],
        "layer_pooling": by_layer,
    }
    layers = len([name for name in features if name.endswith("_mean")])
    final = by_layer[f"layer_{layers}_mean"]
    return {
        "student": str(student_path),
        "student_sha256": hashlib.sha256(student_path.read_bytes()).hexdigest(),
        "train_meta": train_meta,
        "test_meta": test_meta,
        "feature_shape": list(features[f"layer_{layers}_mean"].shape),
        "pooling": (
            "masked mean over valid, non-special tokens; uncorrupted inputs; frozen eval mode"
        ),
        "layers": layers,
        "labels": list(AG_NEWS_LABELS),
        "accuracy": final["accuracy"],
        "macro_f1": final["macro_f1"],
        "majority_baseline": final["majority_baseline"],
        "train_samples": final["train_samples"],
        "test_samples": final["test_samples"],
        "num_classes": final["num_classes"],
        "layer_pooling": by_layer,
    }


def random_baseline(
    *,
    train: Split,
    test: Split,
    model_config: dict[str, Any],
    seeds: Sequence[int],
    tokenizer_name: str,
    max_length: int,
    batch_size: int,
    device: torch.device,
    input_corruption: str = "clean",
) -> dict[str, Any]:
    """Random-init encoders of the same architecture, one per seed.

    The band, not a single draw, is the reference: a single random initialisation is an
    arbitrary point and cannot support "no trained student beats random init".
    """
    from src.config import ExperimentConfig
    from src.data import get_bert_tokenizer
    from src.model import SmallEncoder

    tokenizer = get_bert_tokenizer(tokenizer_name)
    cfg = ExperimentConfig.from_dict({"model": model_config}).model
    train_texts, train_labels, train_meta = train
    test_texts, test_labels, test_meta = test
    accuracies: list[float] = []
    macro_f1s: list[float] = []
    feature_shape: list[int] | None = None
    for seed in seeds:
        torch.manual_seed(int(seed))
        model = SmallEncoder(cfg).to(device).eval()
        features = extract_feature_matrix(
            model,
            tokenizer,
            train_texts,
            max_length=max_length or cfg.max_length,
            batch_size=batch_size,
            device=device,
            special_token_ids=tokenizer.all_special_ids,
            layer_grid=False,
            input_corruption=input_corruption,
            corruption_seed=int(seeds[0]),
        )
        test_features = extract_feature_matrix(
            model,
            tokenizer,
            test_texts,
            max_length=max_length or cfg.max_length,
            batch_size=batch_size,
            device=device,
            special_token_ids=tokenizer.all_special_ids,
            layer_grid=False,
            input_corruption=input_corruption,
            corruption_seed=int(seeds[0]),
        )
        name = f"layer_{cfg.layers}_mean"
        feature_shape = list(features[name].shape)
        result = probe_features(
            features[name],
            np.asarray(train_labels),
            test_features[name],
            np.asarray(test_labels),
            seed=int(seeds[0]),
        )
        accuracies.append(result["accuracy"])
        macro_f1s.append(result["macro_f1"])
    accuracy = summarise(accuracies)
    macro_f1 = summarise(macro_f1s)
    return {
        "student": "random-initialised SmallEncoder (same architecture)",
        "student_sha256": None,
        "seeds": [int(seed) for seed in seeds],
        "accuracies": accuracies,
        "macro_f1s": macro_f1s,
        "accuracy": accuracy["mean"],
        "accuracy_sd": accuracy["sd"],
        "accuracy_min": accuracy["min"],
        "accuracy_max": accuracy["max"],
        "macro_f1": macro_f1["mean"],
        "macro_f1_sd": macro_f1["sd"],
        "sd_definition": "population standard deviation over the seed list",
        "majority_baseline": float(
            max(test_labels.count(label) for label in set(test_labels)) / len(test_labels)
        ),
        "train_meta": train_meta,
        "test_meta": test_meta,
        "feature_shape": feature_shape,
        "input_corruption": input_corruption,
        "pooling": (
            "masked mean over valid, non-special tokens; final layer; frozen eval mode"
        ),
    }


def lexical_baseline(
    *,
    train: Split,
    test: Split,
    seed: int,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_features: int | None = 200_000,
) -> dict[str, Any]:
    """TF-IDF + logistic regression on the identical splits: the lexical floor."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    train_texts, train_labels, train_meta = train
    test_texts, test_labels, test_meta = test
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range, min_df=min_df, max_features=max_features, sublinear_tf=True
    )
    train_matrix = vectorizer.fit_transform(train_texts)
    test_matrix = vectorizer.transform(test_texts)
    classifier = LogisticRegression(C=1.0, max_iter=1000, random_state=seed, solver="lbfgs")
    classifier.fit(train_matrix, train_labels)
    predictions = classifier.predict(test_matrix)
    return {
        "model": "TF-IDF + logistic regression",
        "ngram_range": list(ngram_range),
        "min_df": min_df,
        "num_features": int(len(vectorizer.vocabulary_)),
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "macro_f1": float(f1_score(test_labels, predictions, average="macro")),
        "majority_baseline": float(
            max(test_labels.count(label) for label in set(test_labels)) / len(test_labels)
        ),
        "train_meta": train_meta,
        "test_meta": test_meta,
        "fit_on": "probe-train only; test split used once",
    }


class _HiddenStateAdapter(torch.nn.Module):
    """Expose a Hugging Face encoder through the project's encoder output surface."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor, return_hidden_states: bool = False
    ) -> SimpleNamespace:
        del return_hidden_states
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Hugging Face includes the embedding output at index 0; the project's convention is
        # post-encoder layers only, so it is dropped here rather than downstream.
        return SimpleNamespace(
            last_hidden_state=output.last_hidden_state,
            hidden_states=tuple(output.hidden_states[1:]),
        )


def pretrained_baseline(
    *,
    model_name: str,
    train: Split,
    test: Split,
    max_length: int,
    batch_size: int,
    device: torch.device,
    seed: int,
    input_corruption: str = "clean",
    layer_grid: bool = False,
) -> dict[str, Any]:
    """A frozen pretrained encoder through the identical probe: the instrument ceiling.

    This is an **evaluation diagnostic**. Its weights never enter the project encoder and it
    is never a project condition. It answers one question: can this readout detect features
    that are known to be good? If it cannot, no student comparison below it is interpretable.
    """
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name).to(device).eval()
    encoder.requires_grad_(False)
    model = _HiddenStateAdapter(encoder)
    train_texts, train_labels, train_meta = train
    test_texts, test_labels, test_meta = test
    features = extract_feature_matrix(
        model,
        tokenizer,
        train_texts,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
        special_token_ids=tokenizer.all_special_ids,
        layer_grid=layer_grid,
        input_corruption=input_corruption,
        corruption_seed=seed,
    )
    test_features = extract_feature_matrix(
        model,
        tokenizer,
        test_texts,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
        special_token_ids=tokenizer.all_special_ids,
        layer_grid=layer_grid,
        input_corruption=input_corruption,
        corruption_seed=seed,
    )
    by_layer = {
        name: probe_features(
            features[name],
            np.asarray(train_labels),
            test_features[name],
            np.asarray(test_labels),
            seed=seed,
        )
        for name in features
    }
    layers = len([name for name in features if name.endswith("_mean")])
    final = by_layer[f"layer_{layers}_mean"]
    return {
        "role": "positive control: evaluation instrument only, never a project condition",
        "model": model_name,
        "layers": layers,
        "feature_shape": list(features[f"layer_{layers}_mean"].shape),
        "input_corruption": input_corruption,
        "accuracy": final["accuracy"],
        "macro_f1": final["macro_f1"],
        "majority_baseline": final["majority_baseline"],
        "train_meta": train_meta,
        "test_meta": test_meta,
        "layer_pooling": by_layer,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen AG News linear probe for student exports.")
    parser.add_argument("--student", action="append", help="student.pt (repeatable)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-limit", type=int, default=4000)
    parser.add_argument("--test-limit", type=int, default=1000)
    parser.add_argument("--tokenizer", default="bert-base-uncased")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--split-strategy",
        default="stratified",
        choices=("stratified", "head"),
        help="head reproduces the earlier first-N rows for comparison",
    )
    parser.add_argument(
        "--input-corruption",
        default="clean",
        choices=("clean", "mask"),
        help="probe clean text, or deterministically masked text, to test the "
        "train/evaluation mismatch",
    )
    parser.add_argument(
        "--pretrained-baseline",
        help="positive control: a frozen pretrained encoder through the identical probe. "
        "Evaluation instrument only; never a project condition.",
    )
    parser.add_argument("--random-baseline", action="store_true")
    parser.add_argument("--baseline-seeds", type=int, default=10)
    parser.add_argument("--no-lexical-baseline", action="store_true")
    parser.add_argument("--no-layer-grid", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if not args.student and not args.pretrained_baseline:
        raise SystemExit("run_probe: pass --student and/or --pretrained-baseline")
    if args.random_baseline and not args.student:
        raise SystemExit("run_probe: --random-baseline needs at least one --student")
    if args.random_baseline and args.baseline_seeds < 1:
        raise SystemExit("--baseline-seeds must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda"
        if (args.device == "auto" and torch.cuda.is_available())
        else ("cpu" if args.device == "auto" else args.device)
    )
    _set_seed(args.seed)

    train, test = load_splits(
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        seed=args.seed,
        strategy=args.split_strategy,
    )
    report: dict[str, Any] = {
        "schema_version": 2,
        "device": str(device),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seed": args.seed,
        "probe": "frozen linear probe, StandardScaler + LogisticRegression fit on train split only",
        "split_strategy": args.split_strategy,
        "input_corruption": args.input_corruption,
        "students": {},
        "not_run": [
            "hyperparameter search on the probe (deliberately fixed at C=1.0, max_iter=1000)",
            "larger probe splits",
        ],
    }

    for path in args.student or []:
        student = Path(path)
        if not student.is_file():
            raise SystemExit(f"run_probe: missing student export {student}")
        print(f"probing {student} on {device} ...", flush=True)
        report["students"][student.parent.name or student.stem] = probe_student(
            student,
            train=train,
            test=test,
            tokenizer_name=args.tokenizer,
            max_length=args.max_length,
            batch_size=args.batch_size,
            seed=args.seed,
            device=device,
            layer_grid=not args.no_layer_grid,
            input_corruption=args.input_corruption,
        )

    if not args.no_lexical_baseline:
        print("fitting TF-IDF lexical baseline ...", flush=True)
        report["lexical_baseline"] = lexical_baseline(train=train, test=test, seed=args.seed)

    if args.pretrained_baseline:
        print(f"probing positive control {args.pretrained_baseline} ...", flush=True)
        report["positive_control"] = pretrained_baseline(
            model_name=args.pretrained_baseline,
            train=train,
            test=test,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed,
            input_corruption=args.input_corruption,
            layer_grid=not args.no_layer_grid,
        )

    if args.random_baseline:
        first = next(iter(report["students"].values()))
        model_config = first.get("model_config")
        if model_config is None:
            payload = torch.load(args.student[0], map_location="cpu", weights_only=True)
            model_config = payload["model_config"]
        seeds = list(range(args.seed, args.seed + args.baseline_seeds))
        print(f"probing {len(seeds)} random-init baselines ...", flush=True)
        report["random_baseline"] = random_baseline(
            train=train,
            test=test,
            model_config=model_config,
            seeds=seeds,
            tokenizer_name=args.tokenizer,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=device,
            input_corruption=args.input_corruption,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary: dict[str, Any] = {
        name: {
            "accuracy": value.get("accuracy"),
            "macro_f1": value.get("macro_f1"),
            "majority_baseline": value.get("majority_baseline"),
        }
        for name, value in report["students"].items()
    }
    if "lexical_baseline" in report:
        summary["lexical_baseline"] = {
            key: report["lexical_baseline"].get(key) for key in ("accuracy", "macro_f1")
        }
    if "positive_control" in report:
        summary["positive_control"] = {
            key: report["positive_control"].get(key) for key in ("accuracy", "macro_f1")
        }
    if "random_baseline" in report:
        summary["random_baseline"] = {
            key: report["random_baseline"].get(key)
            for key in ("accuracy", "accuracy_sd", "accuracy_min", "accuracy_max")
        }
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
