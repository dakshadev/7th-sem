"""Tests for the P3 controls: fixed/noise targets, matched head, target geometry."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts import target_geometry
from src.config import (
    ExperimentConfig,
    ModelConfig,
    ObjectiveConfig,
    OptimizerConfig,
    RunConfig,
    TeacherConfig,
)
from src.objectives import MLMHead
from src.train import Trainer


def control_config(mode: str) -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(
            vocab_size=32,
            max_length=8,
            width=16,
            layers=2,
            heads=4,
            ffn_width=32,
            dropout=0.0,
            special_token_ids=(0, 1, 2, 3),
        ),
        objective=ObjectiveConfig(
            variant="latent_a", predictor_dropout=0.0
        ),
        teacher=TeacherConfig(ema_decay=0.5, mode=mode, target_norm="none"),
        optimizer=OptimizerConfig(learning_rate=1e-2, warmup_updates=0, max_updates=8),
        run=RunConfig(seed=7, token_budget=1000, checkpoint_every_updates=1),
    )


def batch() -> dict[str, torch.Tensor]:
    ids = torch.tensor([[1, 8, 9, 10, 2, 0], [1, 11, 12, 13, 2, 0]])
    selected = torch.tensor(
        [[False, True, False, True, False, False], [False, True, False, True, False, False]]
    )
    masked = ids.clone()
    masked[selected] = 3
    return {
        "input_ids": ids,
        "masked_input_ids": masked,
        "attention_mask": ids.ne(0),
        "prediction_mask": selected,
    }


@pytest.mark.parametrize("mode", ["constant", "noise"])
def test_control_targets_need_no_teacher_and_train(mode: str) -> None:
    trainer = Trainer(control_config(mode))
    assert trainer.teacher is None
    result = trainer.step(batch())
    assert result.updated and result.loss is not None
    assert torch.isfinite(torch.tensor(result.loss))
    assert trainer.checkpoint_state()["teacher"] is None


def test_constant_targets_are_zero_and_noise_is_scale_matched() -> None:
    constant = Trainer(control_config("constant"))
    zeros = constant._latent_targets(batch(), None)
    assert zeros.shape == (4, 16)
    assert torch.count_nonzero(zeros) == 0

    noise = Trainer(control_config("noise"))
    first = noise._latent_targets(batch(), None)
    second = noise._latent_targets(batch(), None)
    assert first.shape == (4, 16)
    assert first.std() > 0
    assert not torch.allclose(first, second)


def test_reduced_mlm_head_is_smaller_and_exact() -> None:
    full = sum(item.numel() for item in MLMHead(256, 30522).parameters())
    small = sum(item.numel() for item in MLMHead(256, 30522, bottleneck=64).parameters())
    assert full == 7_910_458
    assert small == 2_066_682
    assert small < full


def test_mlm_head_width_is_rejected_for_latent_objectives() -> None:
    with pytest.raises(ValueError, match="mlm_head_width"):
        ObjectiveConfig(kind="latent", mlm_head_width=64)


def test_target_geometry_from_a_checkpoint(tmp_path: Path) -> None:
    trainer = Trainer(control_config("ema"))
    checkpoint = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(checkpoint)
    batches = tmp_path / "batches.pt"
    torch.save([batch(), batch()], batches)

    report = target_geometry.build_report(
        checkpoint, batches, max_batches=2, device=torch.device("cpu")
    )
    assert report["counts"]["targets"] == 8
    assert report["counts"]["width"] == 16
    assert report["geometry"]["num_samples"] == 8
    assert "is_degenerate" in report["geometry"]
    assert report["target_statistics"]["mean_norm"] > 0
    assert report["checkpoint"]["sha256"]
    assert report["artifact"]["sha256"]


def test_target_geometry_refuses_a_checkpoint_without_a_teacher(tmp_path: Path) -> None:
    trainer = Trainer(control_config("self"))
    checkpoint = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(checkpoint)
    batches = tmp_path / "batches.pt"
    torch.save([batch()], batches)
    with pytest.raises(ValueError, match="no teacher"):
        target_geometry.build_report(
            checkpoint, batches, max_batches=1, device=torch.device("cpu")
        )


def test_control_config_is_a_valid_experiment() -> None:
    config = control_config("noise")
    assert config.teacher.mode == "noise"
    assert ExperimentConfig.from_dict(config.to_dict()) == config
