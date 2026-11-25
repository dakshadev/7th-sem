"""Adapter-neutral fp32 runner for MLM C and latent recipes A/B."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .config import ExperimentConfig, load_config
from .model import SmallEncoder
from .objectives import (
    AttentionPredictor,
    LinearPredictor,
    LossResult,
    MLMHead,
    collapse_regulariser,
    final_target,
    latent_loss,
    normalized_top2_target,
    selected_mlm_loss,
)

CHECKPOINT_SCHEMA = 1
Batch = Mapping[str, Tensor]


@dataclass
class Counters:
    updates: int = 0
    nonpadding_tokens: int = 0
    masked_targets: int = 0
    skipped_batches: int = 0
    training_seconds: float = 0.0
    validation_seconds: float = 0.0


@dataclass(frozen=True)
class StepResult:
    loss: float | None
    targets: int
    updated: bool
    nonpadding_tokens: int
    # Present only when an anti-collapse term is active, so the metrics log shows how much of
    # the gradient came from the statistics term rather than the objective.
    regulariser: float | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: Mapping[str, Any]) -> None:
    """Restore byte tensors on their required CPU side, independent of trainer device."""
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"].detach().cpu())
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.detach().cpu() for item in state["torch_cuda"]])


class StatefulBatchSequence(Iterable[Batch]):
    """Reference in-memory sequence that advances only after a processed batch."""

    def __init__(self, batches: list[dict[str, Tensor]]) -> None:
        self.batches = batches
        self.position = 0
        self._pending = False

    def __iter__(self) -> Iterator[Batch]:
        while self.position < len(self.batches):
            if self._pending:
                self.position += 1
                self._pending = False
                if self.position >= len(self.batches):
                    break
            self._pending = True
            yield self.batches[self.position]

    def commit(self) -> None:
        """Acknowledge a successfully processed batch without advancing on failure."""
        if not self._pending or self.position >= len(self.batches):
            raise ValueError("cannot commit past end of batch sequence")
        self.position += 1
        self._pending = False

    def state_dict(self) -> dict[str, int]:
        return {"position": self.position}

    def load_state_dict(self, state: Mapping[str, int]) -> None:
        position = state["position"]
        if not 0 <= position <= len(self.batches):
            raise ValueError("invalid batch sequence position")
        self.position = position
        self._pending = False


class Trainer:
    def __init__(self, config: ExperimentConfig, device: str | torch.device | None = None) -> None:
        self.config = config
        self.device = torch.device(device or config.run.device)
        set_seed(config.run.seed)
        self.student = SmallEncoder(config.model).to(self.device)
        self.mlm_head: MLMHead | None = None
        self.predictor: nn.Module | None = None
        self.teacher: SmallEncoder | None = None
        if config.objective.kind == "mlm":
            self.mlm_head = MLMHead(config.model.width, config.model.vocab_size).to(self.device)
            modules: list[nn.Module] = [self.student, self.mlm_head]
        else:
            if config.teacher.mode == "ema":
                teacher = SmallEncoder(config.model).to(self.device)
                teacher.load_state_dict(self.student.state_dict())
                teacher.requires_grad_(False)
                teacher.eval()
                self.teacher = teacher
            self.predictor = (
                AttentionPredictor(config.model, config.objective.predictor_dropout)
                if config.objective.predictor == "attention"
                else LinearPredictor(config.model.width)
            ).to(self.device)
            modules = [self.student, self.predictor]
        parameters = [parameter for module in modules for parameter in module.parameters()]
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=config.optimizer.learning_rate,
            betas=config.optimizer.betas,
            weight_decay=config.optimizer.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self._schedule_scale)
        self.counters = Counters()
        self._last_regulariser: float | None = None

    def _schedule_scale(self, update: int) -> float:
        warmup, maximum = self.config.optimizer.warmup_updates, self.config.optimizer.max_updates
        if warmup and update < warmup:
            return (update + 1) / warmup
        return max(0.0, (maximum - update) / max(1, maximum - warmup))

    def _move_and_validate(self, batch: Batch) -> dict[str, Tensor]:
        required = {"input_ids", "masked_input_ids", "attention_mask", "prediction_mask"}
        if set(batch) != required:
            raise ValueError(f"batch keys must be exactly {sorted(required)}")
        local = {key: value.to(self.device) for key, value in batch.items()}
        ids, masked = local["input_ids"], local["masked_input_ids"]
        attention, selected = local["attention_mask"], local["prediction_mask"]
        if ids.dtype != torch.long or masked.dtype != torch.long:
            raise ValueError("input_ids and masked_input_ids must be int64")
        if attention.dtype != torch.bool or selected.dtype != torch.bool:
            raise ValueError("attention_mask and prediction_mask must be boolean")
        if ids.ndim != 2 or ids.shape[0] == 0 or ids.shape[1] == 0:
            raise ValueError("batch must have nonempty [batch, length]")
        if any(item.shape != ids.shape for item in local.values()):
            raise ValueError("all batch fields must share [batch, length]")
        if not attention.any(dim=1).all():
            raise ValueError("empty examples are invalid")
        for name, values in (("input_ids", ids), ("masked_input_ids", masked)):
            if values.min().item() < 0 or values.max().item() >= self.config.model.vocab_size:
                raise ValueError(f"{name} has IDs outside vocabulary")
        if (masked[~selected] != ids[~selected]).any():
            raise ValueError("unselected masked_input_ids must equal input_ids")
        if (selected & ~attention).any():
            raise ValueError("prediction_mask cannot include padding")
        specials = torch.zeros_like(selected)
        for token_id in self.config.model.special_token_ids:
            specials |= ids.eq(token_id)
        if (selected & specials).any():
            raise ValueError("prediction_mask cannot include special tokens")
        return local

    def _latent_targets(self, batch: dict[str, Tensor], student_output) -> Tensor:
        """Teacher targets for the latent objective.

        In 'ema' mode a separate, slowly moving copy of the student supplies them. In 'self'
        mode the student's own forward pass does, detached: that is the ablation which tests
        whether the EMA teacher is doing any work.
        """
        want_hidden = self.config.objective.target == "top2_normalized"
        if self.config.teacher.mode == "ema":
            assert self.teacher is not None
            self.teacher.eval()
            with torch.no_grad():
                source = self.teacher(
                    batch["input_ids"], batch["attention_mask"], return_hidden_states=want_hidden
                )
        else:
            with torch.no_grad():
                source = self.student(
                    batch["input_ids"], batch["attention_mask"], return_hidden_states=want_hidden
                )
        if self.config.objective.target == "final":
            return final_target(source.last_hidden_state, batch["prediction_mask"])
        return normalized_top2_target(source.hidden_states, batch["prediction_mask"])

    def _loss(self, batch: dict[str, Tensor]):
        student_output = self.student(batch["masked_input_ids"], batch["attention_mask"])
        if self.mlm_head is not None:
            outcome = selected_mlm_loss(
                self.mlm_head(student_output.last_hidden_state),
                batch["input_ids"],
                batch["prediction_mask"],
            )
        else:
            assert self.predictor is not None
            targets = self._latent_targets(batch, student_output)
            predictions = self.predictor(
                student_output.last_hidden_state, batch["attention_mask"]
            )
            outcome = latent_loss(
                predictions,
                targets,
                batch["prediction_mask"],
                self.config.objective.loss,
                self.config.objective.smooth_l1_beta,
            )
        return self._add_regulariser(outcome, student_output, batch)

    def _add_regulariser(self, outcome: LossResult, student_output, batch: dict[str, Tensor]):
        """Add the configured anti-collapse term over all non-padding positions.

        Returns the same LossResult type the caller expects; the scalar value is stashed on the
        trainer so ``step`` can put it in the metrics row without changing the objective API.
        """
        self._last_regulariser = None
        settings = self.config.regulariser
        if settings.kind == "none" or outcome.loss is None:
            return outcome
        embeddings = student_output.last_hidden_state[batch["attention_mask"]]
        term = collapse_regulariser(
            embeddings,
            settings.kind,
            variance_floor=settings.variance_floor,
            variance_weight=settings.variance_weight,
            covariance_weight=settings.covariance_weight,
            diagonal_weight=settings.diagonal_weight,
            eps=settings.eps,
        )
        if term is None:
            return outcome
        self._last_regulariser = float(term.detach().item())
        return LossResult(loss=outcome.loss + settings.weight * term, targets=outcome.targets)

    def step(self, batch: Batch, training: bool = True) -> StepResult:
        local = self._move_and_validate(batch)
        tokens = int(local["attention_mask"].sum().item())
        if not bool(local["prediction_mask"].any()):
            if training:
                self.counters.nonpadding_tokens += tokens
                self.counters.skipped_batches += 1
            return StepResult(None, 0, False, tokens)
        for module in (self.student, self.mlm_head, self.predictor):
            if module is not None:
                module.train(training)
        with torch.enable_grad() if training else torch.no_grad():
            outcome = self._loss(local)
        assert outcome.loss is not None
        if not torch.isfinite(outcome.loss):
            raise FloatingPointError("non-finite objective loss")
        if not training:
            return StepResult(
                float(outcome.loss.item()), outcome.targets, False, tokens, self._last_regulariser
            )
        self.optimizer.zero_grad(set_to_none=True)
        outcome.loss.backward()
        parameters = [item for group in self.optimizer.param_groups for item in group["params"]]
        torch.nn.utils.clip_grad_norm_(parameters, self.config.optimizer.grad_clip_norm)
        if not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in parameters
        ):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("non-finite gradients")
        self.optimizer.step()
        self.scheduler.step()
        self._ema_update()
        self.counters.updates += 1
        self.counters.nonpadding_tokens += tokens
        self.counters.masked_targets += outcome.targets
        return StepResult(
            float(outcome.loss.item()), outcome.targets, True, tokens, self._last_regulariser
        )

    def validate(self, batch: Batch) -> StepResult:
        """Validation cannot consume caller-visible random state or mutate counters."""
        state = _rng_state()
        try:
            return self.step(batch, training=False)
        finally:
            _restore_rng(state)

    def _current_ema_decay(self) -> float:
        """Constant decay, or a linear ramp to ema_decay_end across max_updates."""
        start = self.config.teacher.ema_decay
        end = self.config.teacher.ema_decay_end
        if end is None:
            return start
        progress = min(1.0, self.counters.updates / max(1, self.config.optimizer.max_updates))
        return start + (end - start) * progress

    def _ema_update(self) -> None:
        if self.teacher is None:
            return
        decay = self._current_ema_decay()
        with torch.no_grad():
            for teacher, student in zip(
                self.teacher.parameters(), self.student.parameters(), strict=True
            ):
                teacher.mul_(decay).add_(student, alpha=1 - decay)
        self.teacher.eval()

    def checkpoint_state(self, data_state: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return {
            "schema_version": CHECKPOINT_SCHEMA,
            "config": self.config.to_dict(),
            "student": self.student.state_dict(),
            "mlm_head": None if self.mlm_head is None else self.mlm_head.state_dict(),
            "predictor": None if self.predictor is None else self.predictor.state_dict(),
            "teacher": None if self.teacher is None else self.teacher.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "counters": asdict(self.counters),
            "rng": _rng_state(),
            "data_state": dict(data_state) if data_state is not None else None,
        }

    def save_checkpoint(
        self, path: str | Path, data_state: Mapping[str, Any] | None = None
    ) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(self.checkpoint_state(data_state), temporary)
        temporary.replace(destination)

    def load_checkpoint(self, path: str | Path) -> Mapping[str, Any] | None:
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        if state.get("schema_version") != CHECKPOINT_SCHEMA:
            raise ValueError("unsupported checkpoint schema")
        if _resume_semantics(state.get("config")) != _resume_semantics(self.config.to_dict()):
            raise ValueError("checkpoint config is incompatible with this trainer")
        self.student.load_state_dict(state["student"])
        if self.mlm_head is not None:
            self.mlm_head.load_state_dict(state["mlm_head"])
        if self.predictor is not None:
            self.predictor.load_state_dict(state["predictor"])
        if self.teacher is not None:
            self.teacher.load_state_dict(state["teacher"])
            self.teacher.requires_grad_(False)
            self.teacher.eval()
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.counters = Counters(**state["counters"])
        _restore_rng(state["rng"])
        return state["data_state"]

    def export_student(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "model_config": asdict(self.config.model),
                "student": self.student.state_dict(),
            },
            destination,
        )


def _resume_semantics(config: Any) -> Any:
    """Runtime output location/device/checkpoint cadence may change across resume."""
    if not isinstance(config, dict):
        return config
    value = json.loads(json.dumps(config))
    if isinstance(value.get("run"), dict):
        for key in ("device", "output_dir", "checkpoint_every_updates"):
            value["run"].pop(key, None)
    return value


def load_exported_student(path: str | Path, device: str | torch.device = "cpu") -> SmallEncoder:
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported student export schema")
    model_config = ExperimentConfig.from_dict({"model": payload["model_config"]}).model
    model = SmallEncoder(model_config).to(device)
    model.load_state_dict(payload["student"])
    return model.eval()


def _source_fingerprint() -> dict[str, Any]:
    try:
        names = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"], text=True
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        names = []
    allowed = ("src/", "tests/", "configs/", "docs/", "pyproject.toml", "README.md", "AGENTS.md")
    files = [
        name.replace("\\", "/") for name in names if name.replace("\\", "/").startswith(allowed)
    ]
    hashes = {}
    for name in sorted(files):
        source = Path(name)
        if source.is_file():
            hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    canonical = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    return {"sha256": hashlib.sha256(canonical).hexdigest(), "files": hashes}


def metadata(config: ExperimentConfig, device: torch.device | None = None) -> dict[str, Any]:
    def git(args: list[str]) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"

    actual_device = str(device or config.run.device)
    device_hash = hashlib.sha256(actual_device.encode()).hexdigest()[:8]
    return {
        "run_id": f"{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}-{device_hash}",
        "started_at": datetime.now(UTC).isoformat(),
        "config": config.to_dict(),
        "git_commit": git(["rev-parse", "HEAD"]),
        "dirty": bool(git(["status", "--porcelain"])),
        "source": _source_fingerprint(),
        "python": sys.version,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": actual_device,
        "cuda_available": torch.cuda.is_available(),
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "precision": "fp32 (AMP unsupported by design)",
        "teacher": {
            "mode": config.teacher.mode,
            "ema_decay": config.teacher.ema_decay,
            "ema_decay_end": config.teacher.ema_decay_end,
            "start_horizon_updates": _ema_horizon(config.teacher.ema_decay),
            "end_horizon_updates": (
                None
                if config.teacher.ema_decay_end is None
                else _ema_horizon(config.teacher.ema_decay_end)
            ),
        },
        "token_counting": "nonpadding original student input tokens; masked targets separately",
        "data_manifest": {"status": "absent"},
        "lifecycle": {"status": "running", "stop_reason": None},
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _append_row(handle: Any, row: Mapping[str, Any]) -> None:
    handle.write(json.dumps(row) + "\n")
    handle.flush()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_batches(
    trainer: Trainer,
    batches: Iterable[Batch],
    output_dir: str | Path,
    *,
    validation_batches: Iterable[Batch] | None = None,
    segment_updates: int | None = None,
    data_state: Callable[[], Mapping[str, Any] | None] | None = None,
    data_manifest: Mapping[str, Any] | None = None,
    remask: Callable[[Batch], Batch] | None = None,
) -> list[dict[str, Any]]:
    """Run a segment; budget checks happen before fetching the next training batch."""
    if segment_updates is not None and segment_updates <= 0:
        raise ValueError("segment_updates must be positive")
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    metadata_path, checkpoint_path = path / "metadata.json", path / "checkpoint.pt"
    started_segment, initial_updates = time.perf_counter(), trainer.counters.updates
    record = metadata(trainer.config, trainer.device)
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        record["run_id"] = previous.get("run_id", record["run_id"])
        record["segments"] = [*previous.get("segments", []), previous.get("lifecycle", {})]
    else:
        record["segments"] = []
    if data_manifest is not None:
        record["data_manifest"] = data_manifest
    _atomic_json(metadata_path, record)
    rows: list[dict[str, Any]] = []
    stop_reason = "exhausted"
    try:
        with (path / "metrics.jsonl").open("a", encoding="utf-8") as log:
            iterator = iter(batches)
            while True:
                if trainer.counters.updates >= trainer.config.optimizer.max_updates:
                    stop_reason = "complete"
                    break
                if trainer.counters.nonpadding_tokens >= trainer.config.run.token_budget:
                    stop_reason = "budget"
                    break
                if (
                    segment_updates is not None
                    and trainer.counters.updates - initial_updates >= segment_updates
                ):
                    stop_reason = "interrupted"
                    break
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                if remask is not None:
                    batch = remask(batch)
                _synchronize(trainer.device)
                started = time.perf_counter()
                result = trainer.step(batch)
                _synchronize(trainer.device)
                elapsed = time.perf_counter() - started
                commit = getattr(batches, "commit", None)
                if callable(commit):
                    commit()
                trainer.counters.training_seconds += elapsed
                row = {
                    "split": "train",
                    "step": trainer.counters.updates,
                    **asdict(result),
                    "elapsed_seconds": elapsed,
                    "cumulative_training_seconds": trainer.counters.training_seconds,
                }
                rows.append(row)
                _append_row(log, row)
                if (
                    result.updated
                    and trainer.counters.updates % trainer.config.run.checkpoint_every_updates == 0
                ):
                    trainer.save_checkpoint(checkpoint_path, data_state() if data_state else None)
            if validation_batches is not None:
                for batch in validation_batches:
                    _synchronize(trainer.device)
                    started = time.perf_counter()
                    result = trainer.validate(batch)
                    _synchronize(trainer.device)
                    elapsed = time.perf_counter() - started
                    trainer.counters.validation_seconds += elapsed
                    row = {
                        "split": "validation",
                        "step": trainer.counters.updates,
                        **asdict(result),
                        "elapsed_seconds": elapsed,
                        "cumulative_validation_seconds": trainer.counters.validation_seconds,
                    }
                    rows.append(row)
                    _append_row(log, row)
    except KeyboardInterrupt:
        trainer.save_checkpoint(checkpoint_path, data_state() if data_state else None)
        record.update(
            {
                "completed_at": datetime.now(UTC).isoformat(),
                "counters": asdict(trainer.counters),
                "lifecycle": {"status": "interrupted", "stop_reason": "interrupted"},
            }
        )
        _atomic_json(metadata_path, record)
        raise
    except BaseException as error:
        trainer.save_checkpoint(checkpoint_path, data_state() if data_state else None)
        record.update(
            {
                "completed_at": datetime.now(UTC).isoformat(),
                "counters": asdict(trainer.counters),
                "exception": repr(error),
                "lifecycle": {"status": "failed", "stop_reason": "failed"},
            }
        )
        _atomic_json(metadata_path, record)
        raise
    trainer.save_checkpoint(checkpoint_path, data_state() if data_state else None)
    record.update(
        {
            "completed_at": datetime.now(UTC).isoformat(),
            "segment_seconds": time.perf_counter() - started_segment,
            "counters": asdict(trainer.counters),
            "outputs": {"checkpoint": str(checkpoint_path), "metrics": str(path / "metrics.jsonl")},
            "lifecycle": {
                "status": "completed"
                if stop_reason in {"complete", "budget", "exhausted"}
                else "interrupted",
                "stop_reason": stop_reason,
            },
        }
    )
    _atomic_json(metadata_path, record)
    return rows


def _ema_horizon(decay: float) -> float | None:
    """Mean lag d/(1-d) of an EMA with the given decay; None when decay is exactly 1."""
    return None if decay >= 1 else decay / (1 - decay)


def _artifact_manifest(path: Path) -> dict[str, Any]:
    return {
        "status": "provided",
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train from a trusted local tensor-batch artifact."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--batches", required=True, help="trusted torch.save list[batch dict] artifact"
    )
    parser.add_argument("--output", help="defaults to run.output_dir in the configuration")
    parser.add_argument("--resume")
    parser.add_argument(
        "--data-manifest", help="optional JSON manifest supplied by the batch producer"
    )
    parser.add_argument("--segment-updates", type=int, help="execution-only bounded segment length")
    args = parser.parse_args()
    config = load_config(args.config)
    batch_path = Path(args.batches)
    loaded = torch.load(batch_path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, list) or not all(isinstance(batch, dict) for batch in loaded):
        raise ValueError("--batches must be a trusted torch list of tensor dictionaries")
    manifest: dict[str, Any] = {
        "batch_artifact": _artifact_manifest(batch_path),
        "external": {"status": "absent"},
    }
    if args.data_manifest:
        external = json.loads(Path(args.data_manifest).read_text(encoding="utf-8"))
        if not isinstance(external, dict):
            raise ValueError("--data-manifest root must be a JSON object")
        manifest["external"] = external
    sequence = StatefulBatchSequence(loaded)
    trainer = Trainer(config)
    if args.resume:
        state = trainer.load_checkpoint(args.resume)
        if state is not None:
            sequence.load_state_dict(state)
    output = Path(args.output or config.run.output_dir)
    run_batches(
        trainer,
        sequence,
        output,
        segment_updates=args.segment_updates,
        data_state=sequence.state_dict,
        data_manifest=manifest,
    )
    trainer.export_student(output / "student.pt")
    metadata_path = output / "metadata.json"
    record = json.loads(metadata_path.read_text(encoding="utf-8"))
    record.setdefault("outputs", {})["student_export"] = str(output / "student.pt")
    _atomic_json(metadata_path, record)


if __name__ == "__main__":
    main()
