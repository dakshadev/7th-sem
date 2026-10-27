"""Bridge Person 1 student exports to Person 2's frozen geometry evaluation.

Person 1 writes ``{"schema_version": 1, "model_config": ..., "student": ...}``.
Person 2's current ``load_student_checkpoint`` only selects ``student_encoder``,
``model`` or ``encoder``, so it does not recognise Person 1's ``"student"`` key.
This module provides a read-only compatibility loader and a payload rewriter;
it deliberately does not edit Person 2's file.
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import Tensor

from .config import ExperimentConfig
from .model import SmallEncoder
from .train import load_exported_student

PERSON2_LOADER_KEYS = ("student_encoder", "model", "encoder")
COMPATIBILITY_NOTE = (
    "Person 2's load_student_checkpoint does not recognise Person 1's 'student' key. "
    "This bridge reads it directly and can emit a 'student_encoder' payload, but the "
    "cross-branch loader fix is still required."
)
DEFAULT_OUTPUT_ROOT = Path("artifacts/person1")


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


def load_evaluate_module() -> Any:
    try:
        return importlib.import_module("src.evaluate")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Person 2's src/evaluate.py is required for geometry evaluation. This branch "
            "does not vendor it; use a checkout that contains src/evaluate.py and install "
            f"its dependencies. Missing module: {error.name or 'unknown'}."
        ) from error


def load_data_module() -> Any:
    try:
        return importlib.import_module("src.data")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Person 2's src/data.py is required to load the tokenizer. This branch does "
            "not vendor it; use a checkout that contains src/data.py and install its "
            f"dependencies. Missing module: {error.name or 'unknown'}."
        ) from error


def _read_payload(source: str | Path | dict[str, Any]) -> tuple[dict[str, Any], str]:
    if isinstance(source, dict):
        return source, "in-memory"
    path = Path(source)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("student artifact must contain a dictionary payload")
    return payload, str(path)


def _extract_payload(payload: dict[str, Any]) -> tuple[str, dict[str, Any], Any, int | None]:
    schema = payload.get("schema_version")
    if isinstance(payload.get("model_config"), dict) and isinstance(payload.get("student"), dict):
        if schema != 1:
            raise ValueError("unsupported student export schema")
        return "student_export", payload["model_config"], payload["student"], schema
    if isinstance(payload.get("config"), dict) and isinstance(payload.get("student"), dict):
        model_config = payload["config"].get("model")
        if not isinstance(model_config, dict):
            raise ValueError("training checkpoint config has no model section")
        return "training_checkpoint", model_config, payload["student"], schema
    for key in PERSON2_LOADER_KEYS:
        if isinstance(payload.get(key), dict):
            model_config = payload.get("model_config") or payload.get("config")
            if isinstance(model_config, dict) and "model" in model_config:
                model_config = model_config["model"]
            if not isinstance(model_config, dict):
                raise ValueError(
                    f"payload uses {key!r} but carries no model configuration to build it"
                )
            return f"person2_key:{key}", model_config, payload[key], schema
    raise ValueError(
        "unsupported student payload: expected Person 1's student export, a Person 1 "
        f"training checkpoint, or one of {list(PERSON2_LOADER_KEYS)} with model config"
    )


def load_student_compat(
    source: str | Path | dict[str, Any],
    device: str | torch.device = "cpu",
) -> tuple[SmallEncoder, dict[str, Any]]:
    """Load a Person 1 export or nested checkpoint without constructing a teacher."""
    payload, label = _read_payload(source)
    source_kind, model_config, student_state, schema = _extract_payload(payload)
    if source_kind == "student_export" and not isinstance(source, dict):
        model = load_exported_student(source, device=device)
        model.requires_grad_(False)
    else:
        model_config_object = ExperimentConfig.from_dict({"model": model_config}).model
        model = SmallEncoder(model_config_object).to(device)
        model.load_state_dict(student_state, strict=True)
        model.eval()
        model.requires_grad_(False)
    info = {
        "source": label,
        "source_kind": source_kind,
        "schema_version": schema,
        "model_config": model_config,
        "device": str(device),
        "compatibility": {
            "person1_export_key": "student",
            "person2_loader_recognized_keys": list(PERSON2_LOADER_KEYS),
            "requires_cross_branch_fix": True,
            "note": COMPATIBILITY_NOTE,
        },
    }
    return model, info


def person2_compatible_state(source: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Return a payload whose key is selected by Person 2's current loader."""
    payload, _ = _read_payload(source)
    _, model_config, student_state, _ = _extract_payload(payload)
    return {
        "schema_version": 1,
        "model_config": model_config,
        "student_encoder": student_state,
        "note": COMPATIBILITY_NOTE,
    }


def geometry_from_embeddings(embeddings: Tensor) -> dict[str, Any]:
    return load_evaluate_module().compute_representation_geometry(embeddings)


def evaluate_geometry(
    model: SmallEncoder,
    tokenizer: Any,
    texts: list[str],
    *,
    max_length: int,
    batch_size: int,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Embed uncorrupted texts with a frozen student and compute sentence geometry."""
    if not texts:
        raise ValueError("at least one held-out text is required")
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    evaluate = load_evaluate_module()
    embeddings: list[Tensor] = []
    nonpadding_tokens = 0
    model.eval()
    with torch.no_grad():
        model.requires_grad_(False)
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            encoding = tokenizer(
                chunk,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].bool().to(device)
            nonpadding_tokens += int(attention_mask.sum().item())
            hidden = model(input_ids, attention_mask).last_hidden_state
            pooled = evaluate.masked_mean_pooling(
                hidden, attention_mask, input_ids, tokenizer.all_special_ids
            )
            embeddings.append(pooled.detach().cpu())
    stacked = torch.cat(embeddings, dim=0)
    return {
        "geometry": evaluate.compute_representation_geometry(stacked),
        "evaluation": {
            "mode": "frozen eval",
            "gradients": "disabled",
            "text_corruption": "none",
            "pooling": "attention-mask-aware mean excluding padding and special tokens",
            "special_token_ids": [int(item) for item in tokenizer.all_special_ids],
            "num_samples": int(stacked.shape[0]),
            "dimension": int(stacked.shape[1]),
            "nonpadding_tokens": nonpadding_tokens,
        },
    }


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested not in {"cpu", "cuda"}:
        raise ValueError("--device must be one of auto, cpu, cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but this interpreter has no CUDA device")
    return requested


def _validate_single_split(split: str) -> None:
    if any(separator in split for separator in (",", "+")):
        raise ValueError("one split only: joining train and validation is not allowed")


def _load_texts(args: argparse.Namespace) -> tuple[list[str], dict[str, Any]]:
    if args.offline_fixture:
        value = json.loads(Path(args.offline_fixture).read_text(encoding="utf-8-sig"))
        if isinstance(value, dict):
            for key in ("texts", "records", "data"):
                if isinstance(value.get(key), list):
                    value = value[key]
                    break
        if not isinstance(value, list):
            raise ValueError("offline fixture must be a JSON list of strings")
        texts = [item.get("text") if isinstance(item, dict) else item for item in value]
        if not all(isinstance(item, str) for item in texts):
            raise ValueError("offline fixture entries must be strings")
        return texts[: args.max_records], {
            "kind": "offline-fixture",
            "fixture": str(args.offline_fixture),
            "requested_max_records": args.max_records,
        }
    _validate_single_split(args.split)
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "the 'datasets' package is required for --dataset mode; use --offline-fixture"
        ) from error
    stream = load_dataset(args.dataset, args.config_name, split=args.split, streaming=True)
    texts: list[str] = []
    seen_rows = 0
    scan_limit = max(args.max_records * 20, args.max_records + 100)
    for row in stream:
        seen_rows += 1
        text = row.get("text") if isinstance(row, dict) else None
        if isinstance(text, str) and text.strip():
            texts.append(text)
            if len(texts) >= args.max_records:
                break
        if seen_rows >= scan_limit:
            break
    return texts, {
        "kind": "datasets-streaming",
        "dataset": args.dataset,
        "config": args.config_name,
        "split": args.split,
        "requested_max_records": args.max_records,
        "seen_rows": seen_rows,
        "revision": "not_pinned_by_streaming_loader",
    }


def evaluate_student(
    source: str | Path | dict[str, Any],
    texts: list[str],
    tokenizer_name: str,
    *,
    max_length: int,
    batch_size: int,
    device: str | torch.device = "cpu",
    source_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model, info = load_student_compat(source, device)
    data_module = load_data_module()
    tokenizer = data_module.get_bert_tokenizer(tokenizer_name)
    measured = evaluate_geometry(
        model,
        tokenizer,
        texts,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
    )
    geometry = measured["geometry"]
    status = "degenerate" if geometry.get("is_degenerate") else "measured"
    return {
        "schema_version": 1,
        "status": status,
        "created_at": datetime.now(UTC).isoformat(),
        "code_commit": git_commit(),
        "student": info,
        "data": {
            "tokenizer": tokenizer_name,
            **(source_metadata or {"kind": "caller-supplied-texts"}),
            "records": len(texts),
        },
        "evaluation": measured["evaluation"],
        "geometry": geometry,
        "not_run": {
            "linear_probe": "not_run: no labelled probe split was supplied",
            "ag_news_probe": "not_run: no labelled probe split was supplied",
            "token_geometry": "not_run: this bridge reports sentence geometry only",
            "pretrained_bert_ambiguity": "not_run: optional analysis not implemented",
            "gpu_benchmark": "not_run: no GPU throughput or memory benchmark was collected",
        },
        "limitations": [
            "Geometry is computed on a bounded held-out sample, not the full corpus.",
            "A random-init or short-pilot student is not evidence of useful transfer.",
            "Person 2's src/evaluate.py and src/data.py must be present; this branch "
            "does not vendor them.",
            COMPATIBILITY_NOTE,
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a Person 1 student export with Person 2's geometry metrics."
    )
    parser.add_argument("--student", required=True)
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--config-name", default="wikitext-103-raw-v1")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--tokenizer", default="bert-base-uncased")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-records", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--offline-fixture")
    parser.add_argument("--out", help="defaults to artifacts/person1/geometry-<student>.json")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_records < 1:
        parser.error("--max-records must be positive")
    try:
        texts, source_metadata = _load_texts(args)
        if not texts:
            raise ValueError("no nonempty held-out texts were selected")
        result = evaluate_student(
            args.student,
            texts,
            args.tokenizer,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=resolve_device(args.device),
            source_metadata=source_metadata,
        )
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        parser.exit(2, f"evaluate_bridge: error: {error}\n")
    out = (
        Path(args.out)
        if args.out
        else DEFAULT_OUTPUT_ROOT / f"geometry-{Path(args.student).stem}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(out)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
