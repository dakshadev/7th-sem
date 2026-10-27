import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from src.config import (
    ExperimentConfig,
    MaskConfig,
    ModelConfig,
    ObjectiveConfig,
    OptimizerConfig,
    RunConfig,
    TeacherConfig,
)
from src.train import StatefulBatchSequence, Trainer, load_exported_student, run_batches


def config(
    variant: str = "mlm", *, dropout: float = 0.0, token_budget: int = 1000
) -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(
            vocab_size=32,
            max_length=8,
            width=16,
            layers=2,
            heads=4,
            ffn_width=32,
            dropout=dropout,
            special_token_ids=(0, 1, 2, 3),
        ),
        objective=ObjectiveConfig(variant=variant, predictor_dropout=dropout),
        mask=MaskConfig(policy="span" if variant == "latent_a" else "independent"),
        optimizer=OptimizerConfig(learning_rate=1e-2, warmup_updates=0, max_updates=8),
        teacher=TeacherConfig(
            ema_decay=0.5,
            target_norm="none" if variant == "latent_a" else "layernorm",
        ),
        run=RunConfig(seed=7, token_budget=token_budget, checkpoint_every_updates=1),
    )


def batch(no_targets: bool = False) -> dict[str, torch.Tensor]:
    ids = torch.tensor([[1, 8, 9, 10, 2, 0], [1, 11, 12, 13, 2, 0]])
    selected = torch.tensor(
        [[False, True, False, True, False, False], [False, True, False, True, False, False]]
    )
    if no_targets:
        selected.zero_()
    masked = ids.clone()
    masked[selected] = 3
    return {
        "input_ids": ids,
        "masked_input_ids": masked,
        "attention_mask": ids.ne(0),
        "prediction_mask": selected,
    }


@pytest.mark.parametrize("variant", ["mlm", "latent_a", "latent_b"])
def test_synthetic_step_is_finite_and_changes_student(variant: str) -> None:
    trainer = Trainer(config(variant))
    before = next(trainer.student.parameters()).detach().clone()
    result = trainer.step(batch())
    assert result.updated and result.loss is not None and torch.isfinite(torch.tensor(result.loss))
    assert not torch.equal(before, next(trainer.student.parameters()))


def test_no_target_validates_but_skips_forwards_and_counts_tokens() -> None:
    trainer = Trainer(config())
    before = copy.deepcopy(trainer.counters)
    trainer.student.forward = lambda *_args, **_kwargs: pytest.fail("no-target must not forward")  # type: ignore[method-assign]
    result = trainer.step(batch(no_targets=True))
    assert not result.updated and result.nonpadding_tokens == 10
    assert trainer.counters.updates == before.updates
    assert trainer.counters.nonpadding_tokens == before.nonpadding_tokens + 10
    assert trainer.counters.skipped_batches == before.skipped_batches + 1


def test_validation_preserves_parameters_counters_and_rng() -> None:
    trainer = Trainer(config(dropout=0.2))
    before = copy.deepcopy(trainer.student.state_dict())
    counters = copy.deepcopy(trainer.counters)
    torch.manual_seed(123)
    expected = torch.rand(4)
    torch.manual_seed(123)
    result = trainer.validate(batch())
    assert result.loss is not None and not result.updated
    torch.testing.assert_close(torch.rand(4), expected)
    assert trainer.counters == counters
    for name, value in before.items():
        torch.testing.assert_close(trainer.student.state_dict()[name], value)


def test_invalid_selected_padding_and_uncorrupted_positions_are_rejected() -> None:
    trainer = Trainer(config())
    invalid_padding = batch()
    invalid_padding["prediction_mask"][0, -1] = True
    with pytest.raises(ValueError, match="padding"):
        trainer.step(invalid_padding)
    invalid_corruption = batch()
    invalid_corruption["masked_input_ids"][0, 2] = 3
    with pytest.raises(ValueError, match="unselected"):
        trainer.step(invalid_corruption)


def test_fixed_batch_overfit_diagnostic_decreases_loss() -> None:
    trainer = Trainer(config())
    losses = [trainer.step(batch()).loss for _ in range(8)]
    assert all(loss is not None for loss in losses)
    assert losses[-1] < losses[0]  # type: ignore[operator]


def test_teacher_initial_copy_no_grads_eval_and_ema_formula() -> None:
    trainer = Trainer(config("latent_a"))
    assert trainer.teacher is not None
    assert not trainer.teacher.training
    assert all(not item.requires_grad for item in trainer.teacher.parameters())
    before = next(trainer.teacher.parameters()).detach().clone()
    torch.testing.assert_close(before, next(trainer.student.parameters()).detach())
    trainer.step(batch())
    torch.testing.assert_close(
        next(trainer.teacher.parameters()), before * 0.5 + next(trainer.student.parameters()) * 0.5
    )
    assert all(item.grad is None for item in trainer.teacher.parameters())


def _assert_tree_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys()
        for key in left:
            _assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left)) and len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            _assert_tree_equal(first, second)
    else:
        assert left == right


def test_dropout_resume_restores_all_state_optimizer_scheduler_and_rng(tmp_path: Path) -> None:
    cfg = config("latent_b", dropout=0.2)
    full = Trainer(cfg)
    full.step(batch())
    full.step(batch())
    expected_rng = torch.rand(5)
    expected = full.checkpoint_state()

    interrupted = Trainer(cfg)
    interrupted.step(batch())
    checkpoint = tmp_path / "resume.pt"
    interrupted.save_checkpoint(checkpoint, {"position": 1})
    resumed = Trainer(cfg)
    assert resumed.load_checkpoint(checkpoint) == {"position": 1}
    resumed.step(batch())
    _assert_tree_equal(expected["student"], resumed.student.state_dict())
    _assert_tree_equal(
        expected["predictor"], resumed.predictor.state_dict() if resumed.predictor else None
    )
    _assert_tree_equal(
        expected["teacher"], resumed.teacher.state_dict() if resumed.teacher else None
    )
    _assert_tree_equal(expected["optimizer"], resumed.optimizer.state_dict())
    _assert_tree_equal(expected["scheduler"], resumed.scheduler.state_dict())
    assert expected["counters"] == resumed.checkpoint_state()["counters"]
    torch.testing.assert_close(torch.rand(5), expected_rng, rtol=0, atol=0)
    assert resumed.teacher is not None and not resumed.teacher.training


def test_export_is_eval_and_has_standalone_tensor_and_output_parity(tmp_path: Path) -> None:
    trainer = Trainer(config(dropout=0.2))
    trainer.step(batch())
    trainer.student.eval()
    ids, attention = batch()["input_ids"], batch()["attention_mask"]
    expected = trainer.student(ids, attention).last_hidden_state
    export = tmp_path / "student.pt"
    trainer.export_student(export)
    standalone = load_exported_student(export)
    assert not standalone.training
    torch.testing.assert_close(standalone(ids, attention).last_hidden_state, expected)


def test_runner_checks_budget_before_fetch_and_no_target_stream_stops(tmp_path: Path) -> None:
    trainer = Trainer(config(token_budget=1))
    trainer.counters.nonpadding_tokens = 1
    fetched = 0

    def source():
        nonlocal fetched
        fetched += 1
        yield batch()

    assert run_batches(trainer, source(), tmp_path / "already-done") == []
    assert fetched == 0

    no_target = Trainer(config(token_budget=12))
    sequence = StatefulBatchSequence([batch(no_targets=True)] * 10)
    rows = run_batches(no_target, sequence, tmp_path / "no-target", data_state=sequence.state_dict)
    assert len(rows) == 2 and no_target.counters.updates == 0
    assert no_target.counters.nonpadding_tokens == 20 and sequence.position == 2


def test_runner_keeps_logs_on_resume_and_noop_resume_does_not_consume(tmp_path: Path) -> None:
    cfg = config(token_budget=10)
    trainer = Trainer(cfg)
    sequence = StatefulBatchSequence([batch(), batch()])
    run_batches(trainer, sequence, tmp_path, data_state=sequence.state_dict)
    assert sequence.position == 1
    resumed = Trainer(cfg)
    state = resumed.load_checkpoint(tmp_path / "checkpoint.pt")
    sequence.load_state_dict(state or {})
    before = sequence.position
    rows = run_batches(resumed, sequence, tmp_path, data_state=sequence.state_dict)
    assert rows == [] and sequence.position == before
    assert len((tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_stateful_batch_sequence_position_advances_only_after_commit() -> None:
    sequence = StatefulBatchSequence([batch(), batch()])
    next(iter(sequence))
    assert sequence.position == 0
    sequence.commit()
    restored = StatefulBatchSequence([batch(), batch()])
    restored.load_state_dict(sequence.state_dict())
    assert len(list(restored)) == 1


def test_cli_smoke_resume_and_metadata(tmp_path: Path) -> None:
    cfg = config(token_budget=100)
    config_path, batches_path, output = (
        tmp_path / "config.json",
        tmp_path / "batches.pt",
        tmp_path / "run",
    )
    config_path.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    torch.save([batch(), batch()], batches_path)
    command = [
        sys.executable,
        "-m",
        "src.train",
        "--config",
        str(config_path),
        "--batches",
        str(batches_path),
        "--output",
        str(output),
        "--segment-updates",
        "1",
    ]
    subprocess.run(command, check=True)
    subprocess.run(command[:-2] + ["--resume", str(output / "checkpoint.pt")], check=True)
    payload = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert payload["data_manifest"]["batch_artifact"]["sha256"]
    assert payload["counters"]["updates"] == 2
    assert (output / "student.pt").is_file()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA-enabled cloud runtime")
def test_cuda_checkpoint_resume_is_portable(tmp_path: Path) -> None:
    """Run automatically in Colab/Kaggle; local CPU development must skip it."""
    cfg = config("latent_b", dropout=0.2)
    trainer = Trainer(cfg, device="cuda")
    trainer.step(batch())
    checkpoint = tmp_path / "cuda-resume.pt"
    trainer.save_checkpoint(checkpoint)
    resumed = Trainer(cfg, device="cuda")
    resumed.load_checkpoint(checkpoint)
    result = resumed.step(batch())
    assert result.updated and resumed.teacher is not None and not resumed.teacher.training
