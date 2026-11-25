"""Measure the geometry of the latent targets themselves.

Student geometry (``src/evaluate_bridge.py``) cannot distinguish a rich target from a
degenerate one: a constant target can train a low-rank student, and a full-rank target can
produce a collapsed one. This script runs the EMA teacher stored in a training checkpoint
over a manifested batch artifact and computes Person 2's representation geometry on the
target vectors at masked positions.

Usage:
    python scripts/target_geometry.py --checkpoint run/checkpoint.pt \
        --batches batches.pt --max-batches 50 --out results/person1/target_geometry.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.config import ExperimentConfig
from src.evaluate import compute_representation_geometry
from src.model import SmallEncoder
from src.objectives import final_target, normalized_top2_target


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_teacher(
    checkpoint_path: Path, device: torch.device
) -> tuple[SmallEncoder, ExperimentConfig, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a dictionary payload")
    config = ExperimentConfig.from_dict(payload["config"])
    state = payload.get("teacher")
    if state is None:
        raise ValueError(
            f"checkpoint has no teacher (teacher mode {config.teacher.mode!r}); only 'ema' "
            "checkpoints carry a target network"
        )
    model = SmallEncoder(config.model).to(device)
    model.load_state_dict(state)
    model.eval()
    model.requires_grad_(False)
    return model, config, payload


def collect_targets(
    model: SmallEncoder,
    config: ExperimentConfig,
    batches: list[dict[str, torch.Tensor]],
    max_batches: int,
    device: torch.device,
) -> np.ndarray:
    if max_batches < 1:
        raise ValueError("max_batches must be positive")
    vectors: list[torch.Tensor] = []
    for batch in batches[:max_batches]:
        local = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            output = model(
                local["input_ids"],
                local["attention_mask"],
                return_hidden_states=config.objective.target == "top2_normalized",
            )
        if config.objective.target == "final":
            target = final_target(output.last_hidden_state, local["prediction_mask"])
        else:
            target = normalized_top2_target(output.hidden_states, local["prediction_mask"])
        if target.numel():
            vectors.append(target.detach().cpu().to(torch.float64))
    if not vectors:
        raise ValueError("no masked targets were found in the selected batches")
    return torch.cat(vectors, dim=0).numpy()


def target_statistics(matrix: np.ndarray) -> dict[str, Any]:
    norms = np.linalg.norm(matrix, axis=1)
    return {
        "mean_norm": float(norms.mean()),
        "std_norm": float(norms.std()),
        "min_norm": float(norms.min()),
        "max_norm": float(norms.max()),
        "mean_per_dimension_std": float(matrix.std(axis=0, ddof=1).mean()),
    }


def build_report(
    checkpoint_path: Path,
    batches_path: Path,
    max_batches: int,
    device: torch.device,
) -> dict[str, Any]:
    model, config, payload = load_teacher(checkpoint_path, device)
    batches = torch.load(batches_path, map_location="cpu", weights_only=True)
    if not isinstance(batches, list) or not all(isinstance(item, dict) for item in batches):
        raise ValueError("--batches must contain a list of tensor dictionaries")
    matrix = collect_targets(model, config, batches, max_batches, device)
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "checkpoint": {"path": str(checkpoint_path), "sha256": sha256(checkpoint_path)},
        "artifact": {"path": str(batches_path), "sha256": sha256(batches_path)},
        "config": {
            "label": config.objective.label,
            "target": config.objective.target,
            "teacher_mode": config.teacher.mode,
            "ema_decay": config.teacher.ema_decay,
            "ema_decay_end": config.teacher.ema_decay_end,
            "checkpoint_schema": payload.get("schema_version"),
        },
        "counts": {
            "batches_available": len(batches),
            "batches_used": min(max_batches, len(batches)),
            "targets": int(matrix.shape[0]),
            "width": int(matrix.shape[1]),
        },
        "geometry": compute_representation_geometry(matrix),
        "target_statistics": target_statistics(matrix),
        "limitations": [
            "Targets are measured on a bounded prefix of one batch artifact, at masked "
            "positions only.",
            "One checkpoint is one training state; this is not a trajectory.",
            "A healthy target rank does not by itself establish useful transfer; pair it "
            "with the frozen probe.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Geometry of the latent targets.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batches", required=True)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--out", required=True)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("target_geometry: CUDA requested but unavailable")
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        ("cpu" if args.device == "auto" else args.device)
    )
    try:
        report = build_report(
            Path(args.checkpoint), Path(args.batches), args.max_batches, device
        )
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        raise SystemExit(f"target_geometry: error: {error}") from error
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(out)
    print(json.dumps(report["geometry"], indent=2))
    print(f"wrote {out}")
    return report


if __name__ == "__main__":
    main()
