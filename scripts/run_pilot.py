"""Run one bounded end-to-end Person 1 pilot and record only measured values."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from scripts.build_batches import (
    BOUNDED_LIMITATION,
    build_batches_from_args,
    manifest_path_for,
    sha256_file,
    write_json_atomic,
)
from src.config import ExperimentConfig, load_config
from src.train import StatefulBatchSequence, Trainer, run_batches

CONFIG_BY_VARIANT = {
    "mlm": "configs/mlm_c.json",
    "latent_a": "configs/latent_a.json",
    "latent_b": "configs/latent_b.json",
    "ablation_d": "configs/ablation_d.json",
    "ablation_e": "configs/ablation_e.json",
    "ablation_f": "configs/ablation_f.json",
    "ablation_g": "configs/ablation_g.json",
}
# CLI name -> the run label recorded in the config. Labels are what appear in results.
VARIANT_LABEL = {
    "mlm": "C",
    "latent_a": "A",
    "latent_b": "B",
    "ablation_d": "D",
    "ablation_e": "E",
    "ablation_f": "F",
    "ablation_g": "G",
}
DEFAULT_OUTPUT_ROOT = Path("artifacts/person1")


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested not in {"cpu", "cuda"}:
        raise ValueError("--device must be one of auto, cpu, cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but this interpreter has no CUDA device")
    return requested


def resolve_config(args: argparse.Namespace) -> tuple[ExperimentConfig, Path]:
    config_path = Path(args.config or CONFIG_BY_VARIANT[args.variant])
    config = load_config(config_path)
    expected = VARIANT_LABEL[args.variant]
    if config.objective.label != expected:
        raise ValueError(
            f"config {config_path} declares label {config.objective.label!r}, not {expected!r} "
            f"for variant {args.variant!r}"
        )
    run = replace(config.run, device=resolve_device(args.device))
    if args.seed is not None:
        # A "run seed" must also seed initialisation, dropout and batch order, not only masking.
        run = replace(run, seed=args.seed)
    if args.max_tokens is not None:
        run = replace(run, token_budget=args.max_tokens)
    return replace(config, run=run), config_path


def _build_namespace(
    args: argparse.Namespace, config: ExperimentConfig, artifact: Path
) -> argparse.Namespace:
    return argparse.Namespace(
        dataset=args.dataset,
        config_name=args.config_name,
        split=args.split,
        tokenizer=args.tokenizer,
        max_length=(args.max_length if args.max_length is not None else config.model.max_length),
        policy=config.mask.policy,
        mask_ratio=config.mask.ratio if args.mask_ratio is None else args.mask_ratio,
        max_records=args.max_records,
        batch_size=args.batch_size,
        seed=config.run.seed if args.seed is None else args.seed,
        out=str(artifact),
        offline_fixture=args.offline_fixture,
    )


def ensure_artifact(
    args: argparse.Namespace, config: ExperimentConfig, output: Path
) -> tuple[Path, dict[str, Any], Path]:
    """Reuse an existing manifested artifact or build a new bounded one."""
    artifact = Path(args.batches) if args.batches else output / "batches.pt"
    manifest_path = manifest_path_for(artifact)
    if artifact.is_file() and not args.rebuild:
        if not manifest_path.is_file():
            raise ValueError(
                f"reusing {artifact} requires a sibling manifest "
                f"({manifest_path.name}); pass --rebuild to regenerate it"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        recorded_policy = manifest.get("masking", {}).get("policy")
        if recorded_policy != config.mask.policy:
            raise ValueError(
                f"artifact masking policy {recorded_policy!r} does not match "
                f"config policy {config.mask.policy!r}"
            )
        recorded_hash = manifest.get("artifact", {}).get("sha256")
        if recorded_hash != sha256_file(artifact):
            raise ValueError("batch artifact sha256 does not match its manifest")
        return artifact, manifest, manifest_path
    manifest = build_batches_from_args(_build_namespace(args, config, artifact))
    return artifact, manifest, manifest_path


def run_pilot(args: argparse.Namespace) -> dict[str, Any]:
    overall_started = time.perf_counter()
    output = Path(args.output) if args.output else DEFAULT_OUTPUT_ROOT / f"pilot-{args.variant}"
    output.mkdir(parents=True, exist_ok=True)
    config, config_path = resolve_config(args)
    artifact, manifest, manifest_path = ensure_artifact(args, config, output)

    loaded = torch.load(artifact, map_location="cpu", weights_only=True)
    if not isinstance(loaded, list) or not all(isinstance(item, dict) for item in loaded):
        raise ValueError("batch artifact must be a trusted list of tensor dictionaries")
    sequence = StatefulBatchSequence(loaded)
    trainer = Trainer(config, device=config.run.device)
    if args.resume:
        state = trainer.load_checkpoint(args.resume)
        if state is not None:
            sequence.load_state_dict(state)
    initial_updates = trainer.counters.updates
    initial_tokens = trainer.counters.nonpadding_tokens
    initial_masked = trainer.counters.masked_targets
    data_manifest = {
        "batch_artifact": {"path": str(artifact), "sha256": sha256_file(artifact)},
        "external": manifest,
    }

    training_started = time.perf_counter()
    rows = run_batches(
        trainer,
        sequence,
        output,
        segment_updates=args.max_updates,
        data_state=sequence.state_dict,
        data_manifest=data_manifest,
    )
    student_path = output / "student.pt"
    trainer.export_student(student_path)
    training_wall_seconds = time.perf_counter() - training_started

    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.setdefault("outputs", {})["student_export"] = str(student_path)
    write_json_atomic(metadata_path, metadata)

    updates = [row for row in rows if row.get("split") == "train"]
    measured = [row for row in updates if row.get("updated") and row.get("loss") is not None]
    summary = {
        "schema_version": 1,
        "status": "completed" if trainer.counters.updates > initial_updates else "no_updates",
        "variant": config.objective.label,
        "config_path": str(config_path),
        "device": str(trainer.device),
        "bounded_by": {
            "segment_updates": args.max_updates,
            "token_budget": config.run.token_budget,
        },
        "measured": {
            "overall_wall_clock_seconds": time.perf_counter() - overall_started,
            "training_wall_clock_seconds": training_wall_seconds,
            "step_seconds": trainer.counters.training_seconds,
            "updates": trainer.counters.updates - initial_updates,
            "nonpadding_tokens": trainer.counters.nonpadding_tokens - initial_tokens,
            "masked_targets": trainer.counters.masked_targets - initial_masked,
            "cumulative_nonpadding_tokens": trainer.counters.nonpadding_tokens,
            "cumulative_masked_targets": trainer.counters.masked_targets,
            "skipped_batches": trainer.counters.skipped_batches,
            "first_train_loss": measured[0]["loss"] if measured else None,
            "last_train_loss": measured[-1]["loss"] if measured else None,
        },
        "stop_reason": metadata.get("lifecycle", {}).get("stop_reason"),
        "metadata_lifecycle": metadata.get("lifecycle"),
        "artifacts": {
            "metadata": str(metadata_path),
            "metrics": str(output / "metrics.jsonl"),
            "checkpoint": str(output / "checkpoint.pt"),
            "student": str(student_path),
            "batch_artifact": str(artifact),
            "batch_manifest": str(manifest_path),
        },
        "data": {
            "batch_artifact_sha256": sha256_file(artifact),
            "source": manifest.get("source"),
            "counts": manifest.get("counts"),
            "limitations": manifest.get("limitations"),
        },
        "not_run": {
            "validation": "not_run: no separate held-out validation split supplied",
            "geometry": "not_run: use src.evaluate_bridge after the pilot",
            "linear_probe": "not_run: labels not supplied",
            "gpu_throughput": (
                "not_run: CPU-only run"
                if str(trainer.device) == "cpu"
                else "not_run: no peak-memory or tokens/second benchmark was collected"
            ),
        },
        "limitations": [
            BOUNDED_LIMITATION,
            "A short pilot cannot establish non-collapse, superiority, or useful transfer.",
            "MLM cross-entropy and latent losses are not directly comparable.",
        ],
    }
    write_json_atomic(output / "pilot.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one bounded Person 1 pilot with the real src.train runner."
    )
    parser.add_argument("--variant", choices=tuple(CONFIG_BY_VARIANT), default="mlm")
    parser.add_argument("--config", help="override the variant's checked-in config")
    parser.add_argument("--output", help="defaults to artifacts/person1/pilot-<variant>")
    parser.add_argument("--batches", help="reuse this artifact instead of building one")
    parser.add_argument("--rebuild", action="store_true", help="rebuild an existing artifact")
    parser.add_argument("--resume", help="training checkpoint to resume")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-updates", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, help="override run.token_budget")
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--config-name", default="wikitext-103-raw-v1")
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer", default="bert-base-uncased")
    parser.add_argument("--max-length", type=int, help="defaults to config model.max_length")
    parser.add_argument("--max-records", type=int, default=500)
    parser.add_argument("--mask-ratio", type=float, help="defaults to config mask.ratio")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, help="defaults to config run.seed")
    parser.add_argument(
        "--offline-fixture",
        help="read texts from a local JSON file instead of the dataset source",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_updates <= 0:
        parser.error("--max-updates must be positive")
    try:
        summary = run_pilot(args)
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        parser.exit(2, f"run_pilot: error: {error}\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
