"""Run a budget sweep with the learning-rate schedule derived from the artifact.

The recorded sweep (`results/person1/sweep_annealed.json`) says optimizer.max_updates was
derived from each token budget so the linear schedule reached zero in every run. That
derivation existed only in prose and in a Kaggle kernel that is not in this repository.
This script makes the rule executable and records it in every run's metadata.

For each (budget, variant, seed):
  1. build or reuse one manifested batch artifact per variant;
  2. derive `max_updates = round(token_budget / tokens_per_update)`, where
     `tokens_per_update` is the whole artifact's mean non-padding tokens per batch
     (`counts.nonpadding_tokens / counts.batches`) unless `--tokens-per-update` is given;
  3. train with that schedule, stopping on the token budget, the schedule, or stream end;
  4. export the student and, with `--probe`, run the frozen linear probe.

This reconstructs the recorded *configuration*, not bit-exact runs. The original used a
fixed 575 tokens/update constant and a segment cap, so a derived run can stop as
`complete` where the recorded 0.69M tier stopped as `interrupted`, and its update count can
differ by a few tenths of a percent. Both facts are recorded in the output.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from scripts.build_batches import (
    build_batches_from_args,
    manifest_path_for,
    sha256_file,
    write_json_atomic,
)
from scripts.run_pilot import CONFIG_BY_VARIANT, VARIANT_LABEL, resolve_device
from src.config import ExperimentConfig, load_config
from src.train import StatefulBatchSequence, Trainer, run_batches


def tokens_per_update(
    batches: list[dict[str, torch.Tensor]], manifest: dict[str, Any] | None = None
) -> float:
    """Mean non-padding tokens per batch over the whole artifact.

    The whole artifact matters: the recorded 0.69M tier stopped at 692,919 tokens / 1,217
    updates (569.4 tokens/update over the consumed prefix), yet its schedule was derived
    from 575, the mean over all 90,000 records. Using the prefix would change the schedule.
    """
    if not batches:
        raise ValueError("artifact has no batches")
    if manifest is not None:
        counts = manifest.get("counts", {})
        if counts.get("batches"):
            return counts["nonpadding_tokens"] / counts["batches"]
    return sum(int(batch["attention_mask"].sum().item()) for batch in batches) / len(batches)


def derive_max_updates(token_budget: int, per_update: float) -> int:
    """Updates needed for the linear schedule to reach zero at the token budget."""
    if token_budget < 1:
        raise ValueError("token budget must be positive")
    if per_update <= 0:
        raise ValueError("tokens per update must be positive")
    return max(1, round(token_budget / per_update))


def config_path_for(args: argparse.Namespace, variant: str) -> Path:
    return Path(args.config_dir) / Path(CONFIG_BY_VARIANT[variant]).name


def artifact_namespace(
    args: argparse.Namespace, config: ExperimentConfig, path: Path
) -> argparse.Namespace:
    return argparse.Namespace(
        dataset=args.dataset,
        config_name=args.config_name,
        split=args.split,
        tokenizer=args.tokenizer,
        max_length=config.model.max_length,
        policy=config.mask.policy,
        mask_ratio=config.mask.ratio,
        mean_span_length=config.mask.span_length,
        max_span_length=config.mask.max_span_length,
        max_records=args.max_records,
        batch_size=args.batch_size,
        seed=args.artifact_seed,
        out=str(path),
        offline_fixture=args.offline_fixture,
    )


def ensure_artifact(
    args: argparse.Namespace, config: ExperimentConfig, path: Path
) -> tuple[Path, dict[str, Any]]:
    manifest_path = manifest_path_for(path)
    if path.is_file() and manifest_path.is_file() and not args.rebuild:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("artifact", {}).get("sha256") != sha256_file(path):
            raise ValueError(f"{path} does not match its manifest; pass --rebuild")
        recorded_policy = manifest.get("masking", {}).get("policy")
        if recorded_policy != config.mask.policy:
            raise ValueError(
                f"artifact masking policy {recorded_policy!r} does not match "
                f"config policy {config.mask.policy!r}"
            )
        return path, manifest
    return path, build_batches_from_args(artifact_namespace(args, config, path))


def train_one(
    *,
    config: ExperimentConfig,
    seed: int,
    budget: int,
    per_update: float,
    per_update_source: str,
    batches: list[dict[str, torch.Tensor]],
    artifact_path: Path,
    manifest: dict[str, Any],
    device: torch.device,
    output_root: Path,
    config_path: Path,
) -> dict[str, Any]:
    max_updates = derive_max_updates(budget, per_update)
    run_dir = output_root / f"b{budget}-s{seed}-{config.objective.label.lower()}"
    run = replace(
        config.run,
        seed=seed,
        token_budget=budget,
        device=str(device),
        output_dir=str(run_dir),
    )
    scheduled = replace(
        config, run=run, optimizer=replace(config.optimizer, max_updates=max_updates)
    )
    sequence = StatefulBatchSequence(batches)
    trainer = Trainer(scheduled, device=device)
    rows = run_batches(
        trainer,
        sequence,
        run_dir,
        data_state=sequence.state_dict,
        data_manifest={
            "batch_artifact": {
                "path": str(artifact_path),
                "sha256": sha256_file(artifact_path),
            },
            "external": manifest,
        },
    )
    student = run_dir / "student.pt"
    trainer.export_student(student)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    measured = [row for row in rows if row.get("split") == "train" and row.get("updated")]
    return {
        "run": run_dir.name,
        "label": config.objective.label,
        "seed": seed,
        "budget_tokens": budget,
        "max_updates": max_updates,
        "tokens_per_update": round(per_update, 6),
        "tokens_per_update_source": per_update_source,
        "updates": trainer.counters.updates,
        "nonpadding_tokens": trainer.counters.nonpadding_tokens,
        "masked_targets": trainer.counters.masked_targets,
        "first_loss": measured[0]["loss"] if measured else None,
        "last_loss": measured[-1]["loss"] if measured else None,
        "seconds": round(trainer.counters.training_seconds, 3),
        "stop_reason": metadata.get("lifecycle", {}).get("stop_reason"),
        "config": str(config_path),
        "student": str(student),
    }


def _parse_ints(text: str, name: str) -> list[int]:
    try:
        values = [int(item) for item in text.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError(f"--{name} must be a comma-separated list of integers") from error
    if not values:
        raise ValueError(f"--{name} must not be empty")
    if any(value < 0 for value in values):
        raise ValueError(f"--{name} values must be nonnegative")
    return values


def _parse_variants(text: str) -> list[str]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("--variants must not be empty")
    unknown = [item for item in values if item not in CONFIG_BY_VARIANT]
    if unknown:
        raise ValueError(f"unknown variants {unknown}; choose from {sorted(CONFIG_BY_VARIANT)}")
    return values


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_records < 1:
        raise ValueError("--max-records must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.tokens_per_update is not None and args.tokens_per_update <= 0:
        raise ValueError("--tokens-per-update must be positive")
    budgets = _parse_ints(args.budgets, "budgets")
    seeds = _parse_ints(args.seeds, "seeds")
    variants = _parse_variants(args.variants)
    device = torch.device(resolve_device(args.device))
    output_root = Path(args.output)
    probe_data = None
    if args.probe:
        from scripts.run_probe import load_splits

        probe_data = load_splits(
            train_limit=args.probe_train,
            test_limit=args.probe_test,
            seed=args.probe_seed,
            strategy=args.probe_split_strategy,
        )

    rows: list[dict[str, Any]] = []
    for variant in variants:
        config_path = config_path_for(args, variant)
        config = load_config(config_path)
        expected = VARIANT_LABEL[variant]
        if config.objective.label != expected:
            raise ValueError(
                f"{config_path_for(args, variant)} declares label "
                f"{config.objective.label!r}, not {expected!r}"
            )
        artifact_path = output_root / f"batches-{config.objective.label}.pt"
        artifact_path, manifest = ensure_artifact(args, config, artifact_path)
        loaded = torch.load(artifact_path, map_location="cpu", weights_only=True)
        if not isinstance(loaded, list) or not all(isinstance(item, dict) for item in loaded):
            raise ValueError("batch artifact must be a trusted list of tensor dictionaries")
        if args.tokens_per_update is None:
            per_update = tokens_per_update(loaded, manifest)
            source = "artifact"
        else:
            per_update = args.tokens_per_update
            source = "cli"
        for seed in seeds:
            for budget in budgets:
                row = train_one(
                    config=config,
                    seed=seed,
                    budget=budget,
                    per_update=per_update,
                    per_update_source=source,
                    batches=loaded,
                    artifact_path=artifact_path,
                    manifest=manifest,
                    device=device,
                    output_root=output_root,
                    config_path=config_path,
                )
                if args.probe and probe_data is not None:
                    from scripts.run_probe import probe_student

                    probed = probe_student(
                        Path(row["student"]),
                        train=probe_data[0],
                        test=probe_data[1],
                        tokenizer_name=args.tokenizer,
                        max_length=config.model.max_length,
                        batch_size=args.probe_batch_size,
                        seed=args.probe_seed,
                        device=device,
                        layer_grid=args.probe_layer_grid,
                    )
                    row["probe_accuracy"] = probed["accuracy"]
                    row["probe_macro_f1"] = probed["macro_f1"]
                rows.append(row)
                print(json.dumps({key: row[key] for key in ("run", "updates", "last_loss")}))

    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "budget sweep with the learning-rate schedule derived from the artifact",
        "created_at": datetime.now(UTC).isoformat(),
        "design": {
            "budgets": budgets,
            "variants": variants,
            "seeds": seeds,
            "tokens_per_update": (
                "cli override" if args.tokens_per_update else "artifact mean, whole artifact"
            ),
            "artifact_policy": "one artifact per variant, reused across seeds and budgets",
            "corpus": {
                "dataset": args.dataset,
                "config": args.config_name,
                "split": args.split,
                "max_records": args.max_records,
                "batch_size": args.batch_size,
                "artifact_seed": args.artifact_seed,
            },
        },
        "runs": rows,
        "limitations": [
            "Derived runs reconstruct the recorded configuration, not bit-exact replicas.",
            "The recorded sweep used a fixed 575 tokens/update and a segment cap; a derived "
            "run may stop as 'complete' where the recorded low tier stopped as 'interrupted'.",
            "Two seeds bound initialisation variance only when the artifact is reused, "
            "because the mask realisation and batch order are fixed.",
            "A bounded sweep is not evidence that one objective beats another at scale.",
        ],
    }
    if args.probe and probe_data is not None and args.random_baseline:
        from scripts.run_probe import random_baseline

        payload = torch.load(rows[0]["student"], map_location="cpu", weights_only=True)
        report["random_baseline"] = random_baseline(
            train=probe_data[0],
            test=probe_data[1],
            model_config=payload["model_config"],
            seeds=list(range(args.probe_seed, args.probe_seed + args.baseline_seeds)),
            tokenizer_name=args.tokenizer,
            max_length=payload["model_config"].get("max_length", 128),
            batch_size=args.probe_batch_size,
            device=device,
        )
    write_json_atomic(args.out, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a budget sweep with max_updates derived from the batch artifact."
    )
    parser.add_argument("--output", required=True, help="root directory for run artifacts")
    parser.add_argument("--out", required=True, help="sweep JSON path")
    parser.add_argument("--budgets", default="700000,5000000")
    parser.add_argument("--variants", default="mlm,latent_a,latent_b")
    parser.add_argument("--seeds", default="17,43")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument(
        "--tokens-per-update",
        type=float,
        help="override the artifact-derived mean; the recorded sweep used 575",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--config-name", default="wikitext-103-raw-v1")
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer", default="bert-base-uncased")
    parser.add_argument("--max-records", type=int, default=90_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--artifact-seed", type=int, default=17)
    parser.add_argument("--offline-fixture")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--random-baseline", action="store_true")
    parser.add_argument("--baseline-seeds", type=int, default=10)
    parser.add_argument("--probe-train", type=int, default=4000)
    parser.add_argument("--probe-test", type=int, default=1000)
    parser.add_argument("--probe-batch-size", type=int, default=32)
    parser.add_argument("--probe-seed", type=int, default=42)
    parser.add_argument(
        "--probe-split-strategy",
        choices=("stratified", "head"),
        default="stratified",
    )
    parser.add_argument(
        "--probe-layer-grid",
        action="store_true",
        help="probe every layer and both poolings; off by default to keep sweeps fast",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_sweep(args)
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        parser.exit(2, f"run_sweep: error: {error}\n")


if __name__ == "__main__":
    main()
