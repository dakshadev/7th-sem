"""Offline, network-free integration tests for the Person 1 <-> Person 2 bridge.

Person 2's modules are not on this branch. The tests install small test doubles
that implement only the documented call surface so the Person 1 paths remain
exercised without copying their production masking or geometry implementation.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from scripts import build_batches, run_pilot
from src.evaluate_bridge import (
    PERSON2_LOADER_KEYS,
    evaluate_geometry,
    evaluate_student,
    geometry_from_embeddings,
    load_student_compat,
    person2_compatible_state,
)
from src.train import load_exported_student

BATCH_KEYS = {"input_ids", "masked_input_ids", "attention_mask", "prediction_mask"}
SPECIAL_IDS = (0, 1, 2, 3)
TEXTS = [
    "the quick brown fox jumps over the lazy dog",
    "a small encoder learns masked latent targets",
    "representation geometry reports effective rank",
    "wikitext is a collection of wikipedia articles",
    "span masking selects short contiguous runs",
    "the pilot records measured tokens and wall clock time",
]


class StubTokenizer:
    vocab_size = 64
    pad_token_id = 0
    mask_token_id = 3
    all_special_ids = list(SPECIAL_IDS)
    name_or_path = "stub-tokenizer"
    init_kwargs = {"_commit_hash": "stub"}

    def __call__(
        self,
        text: str | list[str],
        max_length: int = 128,
        truncation: bool = True,
        padding: str = "max_length",
        return_attention_mask: bool = True,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        del truncation, padding, return_attention_mask, return_tensors
        items = [text] if isinstance(text, str) else text
        rows: list[list[int]] = []
        special_rows: list[list[int]] = []
        attention_rows: list[list[int]] = []
        for item in items:
            words = re.findall(r"[a-z0-9]+", item.lower())
            token_ids = [5 + (sum(word.encode()) % 59) for word in words][: max_length - 2]
            ids = [1, *token_ids, 2][:max_length]
            padding_count = max_length - len(ids)
            rows.append(ids + [0] * padding_count)
            special_rows.append([1] + [0] * (len(ids) - 2) + [1] + [1] * padding_count)
            attention_rows.append([1] * len(ids) + [0] * padding_count)
        return {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.tensor(attention_rows, dtype=torch.long),
            "special_tokens_mask": torch.tensor(special_rows, dtype=torch.long),
        }


def stub_process_single_text(
    text: str,
    tokenizer: StubTokenizer,
    max_length: int = 128,
    mask_ratio: float = 0.15,
    mask_mode: str = "independent",
    generator: torch.Generator | None = None,
    mean_span_length: int = 3,
    max_span_length: int = 5,
) -> dict[str, torch.Tensor]:
    del mean_span_length, max_span_length
    encoded = tokenizer(text, max_length=max_length)
    input_ids = encoded["input_ids"][0]
    attention_mask = encoded["attention_mask"][0].bool()
    special_mask = encoded["special_tokens_mask"][0].bool()
    eligible = attention_mask & ~special_mask
    indices = torch.nonzero(eligible, as_tuple=False).squeeze(-1)
    selected = torch.zeros_like(eligible)
    if indices.numel():
        count = min(indices.numel(), max(1, round(indices.numel() * mask_ratio)))
        order = torch.randperm(indices.numel(), generator=generator)
        if mask_mode == "independent":
            selected[indices[order[:count]]] = True
        elif mask_mode == "span":
            start = int(indices[order[0]])
            for offset in range(count):
                position = start + offset
                if position >= eligible.numel() or not eligible[position]:
                    break
                selected[position] = True
            remaining = count - int(selected.sum())
            if remaining:
                available = indices[~selected[indices]]
                selected[available[:remaining]] = True
        else:
            raise ValueError(f"unsupported mask mode {mask_mode}")
    masked_input_ids = input_ids.clone()
    masked_input_ids[selected] = tokenizer.mask_token_id
    return {
        "input_ids": input_ids,
        "masked_input_ids": masked_input_ids,
        "attention_mask": attention_mask,
        "prediction_mask": selected,
    }


def stub_collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in batch]) for key in BATCH_KEYS}


def stub_masked_mean_pooling(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    input_ids: torch.Tensor,
    special_token_ids: list[int] | set[int],
) -> torch.Tensor:
    special = torch.isin(input_ids, torch.tensor(list(special_token_ids), dtype=input_ids.dtype))
    valid = attention_mask.bool() & ~special
    counts = valid.sum(dim=1, keepdim=True).clamp(min=1)
    return (hidden_states * valid.unsqueeze(-1)).sum(dim=1) / counts


def stub_geometry(embeddings: torch.Tensor | np.ndarray) -> dict[str, Any]:
    if isinstance(embeddings, torch.Tensor):
        matrix = embeddings.detach().cpu().double().numpy()
    else:
        matrix = np.asarray(embeddings, dtype=np.float64)
    samples, dimension = matrix.shape
    if not np.isfinite(matrix).all():
        return {"is_degenerate": True, "status": "nan_or_inf_detected"}
    if samples < 2:
        return {"is_degenerate": True, "status": "insufficient_samples_N_less_than_2"}
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    variance = centered.var(axis=0, ddof=1)
    if float(variance.sum()) < 1e-12:
        return {
            "is_degenerate": True,
            "status": "constant_representations_collapsed",
            "num_samples": samples,
            "dimension": dimension,
            "effective_rank_singular": 1.0,
            "effective_rank_covariance": 1.0,
            "mean_dimension_variance": 0.0,
            "min_dimension_variance": 0.0,
            "max_dimension_variance": 0.0,
            "dead_dimension_count": dimension,
            "dead_dimension_fraction": 1.0,
        }
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    probabilities = singular / singular.sum()
    rank = float(np.exp(-np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)))))
    return {
        "is_degenerate": False,
        "status": "healthy",
        "num_samples": samples,
        "dimension": dimension,
        "effective_rank_singular": rank,
        "effective_rank_covariance": rank,
        "rank_ratio": rank / min(samples, dimension),
        "mean_dimension_variance": float(variance.mean()),
        "min_dimension_variance": float(variance.min()),
        "max_dimension_variance": float(variance.max()),
        "dead_dimension_count": 0,
        "dead_dimension_fraction": 0.0,
        "mean_pairwise_cosine": 0.0,
        "std_pairwise_cosine": 0.0,
        "min_pairwise_cosine": 0.0,
        "max_pairwise_cosine": 0.0,
        "median_pairwise_cosine": 0.0,
        "p10_pairwise_cosine": 0.0,
        "p90_pairwise_cosine": 0.0,
    }


def install_person2_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    def get_tokenizer(_name: str = "stub-tokenizer") -> StubTokenizer:
        return StubTokenizer()

    data_module = types.ModuleType("src.data")
    data_module.get_bert_tokenizer = get_tokenizer  # type: ignore[attr-defined]
    data_module.process_single_text = stub_process_single_text  # type: ignore[attr-defined]
    data_module.collate_batch = stub_collate  # type: ignore[attr-defined]

    evaluate_module = types.ModuleType("src.evaluate")
    evaluate_module.masked_mean_pooling = stub_masked_mean_pooling  # type: ignore[attr-defined]
    evaluate_module.compute_representation_geometry = stub_geometry  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "src.data", data_module)
    monkeypatch.setitem(sys.modules, "src.evaluate", evaluate_module)


def write_tiny_config(tmp_path: Path, variant: str = "mlm") -> Path:
    config = {
        "schema_version": 1,
        "model": {
            "vocab_size": 64,
            "max_length": 16,
            "width": 16,
            "layers": 2,
            "heads": 4,
            "ffn_width": 32,
            "dropout": 0.0,
            "pad_token_id": 0,
            "special_token_ids": list(SPECIAL_IDS),
        },
        "optimizer": {
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "warmup_updates": 0,
            "max_updates": 50,
        },
        "teacher": {
            "ema_decay": 0.5,
            "target_norm": "none" if variant == "latent_a" else "layernorm",
        },
        "mask": {
            "policy": "span" if variant == "latent_a" else "independent",
            "ratio": 0.15,
            "span_length": 3,
        },
        "objective": {"variant": variant, "predictor_dropout": 0.0},
        "run": {
            "seed": 7,
            "token_budget": 100_000,
            "device": "cpu",
            "output_dir": "artifacts/person1/test",
            "checkpoint_every_updates": 1,
        },
    }
    path = tmp_path / f"config-{variant}.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def write_fixture(tmp_path: Path, texts: list[str] | None = None) -> Path:
    path = tmp_path / "texts.json"
    path.write_text(json.dumps({"texts": texts or TEXTS}), encoding="utf-8")
    return path


def run_tiny_pilot(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    output = tmp_path / "pilot"
    summary = run_pilot.main(
        [
            "--variant",
            "mlm",
            "--config",
            str(write_tiny_config(tmp_path)),
            "--output",
            str(output),
            "--offline-fixture",
            str(write_fixture(tmp_path)),
            "--dataset",
            "offline-pilot",
            "--config-name",
            "fixture",
            "--split",
            "train",
            "--tokenizer",
            "stub-tokenizer",
            "--max-length",
            "16",
            "--max-records",
            "4",
            "--batch-size",
            "2",
            "--seed",
            "7",
            "--max-updates",
            "2",
            "--device",
            "cpu",
        ]
    )
    return output, summary


def test_build_batches_writes_contract_artifact_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_person2_stubs(monkeypatch)
    out = tmp_path / "batches.pt"
    manifest = build_batches.main(
        [
            "--dataset",
            "offline-build",
            "--config-name",
            "fixture",
            "--split",
            "train",
            "--tokenizer",
            "stub-tokenizer",
            "--max-length",
            "16",
            "--policy",
            "independent",
            "--mask-ratio",
            "0.15",
            "--max-records",
            "6",
            "--batch-size",
            "2",
            "--seed",
            "5",
            "--out",
            str(out),
            "--offline-fixture",
            str(write_fixture(tmp_path)),
        ]
    )

    assert out.is_file()
    manifest_path = build_batches.manifest_path_for(out)
    assert manifest_path.is_file()
    written = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert written["artifact"]["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert written == manifest
    assert written["source"]["dataset"] == "offline-build"
    assert written["source"]["config"] == "fixture"
    assert written["source"]["split"] == "train"
    assert written["tokenizer"]["id"] == "stub-tokenizer"
    assert written["counts"]["records"] == 6
    assert written["counts"]["batches"] == 3
    assert written["counts"]["nonpadding_tokens"] > 0
    assert written["counts"]["masked_targets"] > 0
    assert "5M-token" in " ".join(written["limitations"])

    batches = torch.load(out, map_location="cpu", weights_only=True)
    assert isinstance(batches, list) and len(batches) == 3
    for batch in batches:
        assert set(batch) == BATCH_KEYS
        assert batch["input_ids"].dtype == torch.int64
        assert batch["masked_input_ids"].dtype == torch.int64
        assert batch["attention_mask"].dtype == torch.bool
        assert batch["prediction_mask"].dtype == torch.bool
        assert batch["input_ids"].shape == (2, 16)
        assert batch["attention_mask"].any(dim=1).all()
        assert not (batch["prediction_mask"] & ~batch["attention_mask"]).any()
        specials = torch.zeros_like(batch["prediction_mask"])
        for token_id in SPECIAL_IDS:
            specials |= batch["input_ids"].eq(token_id)
        assert not (batch["prediction_mask"] & specials).any()
        assert batch["prediction_mask"].any()
        unmasked = ~batch["prediction_mask"]
        torch.testing.assert_close(
            batch["masked_input_ids"][unmasked], batch["input_ids"][unmasked]
        )


def test_build_batches_refuses_joined_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_person2_stubs(monkeypatch)
    with pytest.raises(SystemExit):
        build_batches.main(
            [
                "--split",
                "train+validation",
                "--out",
                str(tmp_path / "invalid.pt"),
            ]
        )


def test_pilot_writes_measured_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_person2_stubs(monkeypatch)
    output, summary = run_tiny_pilot(tmp_path)

    for name in (
        "metadata.json",
        "metrics.jsonl",
        "checkpoint.pt",
        "student.pt",
        "pilot.json",
        "batches.pt",
    ):
        assert (output / name).is_file()

    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["counters"]["updates"] == 2
    assert metadata["counters"]["nonpadding_tokens"] > 0
    assert metadata["data_manifest"]["external"]["artifact"]["sha256"]
    assert metadata["outputs"]["student_export"].endswith("student.pt")

    rows = [
        json.loads(line)
        for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 2
    assert all(row["updated"] and row["loss"] is not None for row in rows)
    assert all(math.isfinite(row["loss"]) for row in rows)

    assert summary["status"] == "completed"
    assert summary["measured"]["updates"] == 2
    assert summary["measured"]["nonpadding_tokens"] == metadata["counters"]["nonpadding_tokens"]
    assert summary["measured"]["first_train_loss"] is not None
    assert summary["measured"]["last_train_loss"] is not None
    assert summary["measured"]["overall_wall_clock_seconds"] > 0
    assert summary["measured"]["step_seconds"] >= 0

    student = load_exported_student(output / "student.pt")
    assert not student.training


def test_bridge_loads_export_and_reports_geometry_and_degeneracy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_person2_stubs(monkeypatch)
    output, _ = run_tiny_pilot(tmp_path)

    result = evaluate_student(
        output / "student.pt",
        TEXTS,
        "stub-tokenizer",
        max_length=16,
        batch_size=2,
        device="cpu",
        source_metadata={"kind": "test"},
    )
    assert result["status"] == "measured"
    assert result["geometry"]["is_degenerate"] is False
    for key in (
        "effective_rank_singular",
        "effective_rank_covariance",
        "mean_dimension_variance",
        "mean_pairwise_cosine",
    ):
        assert math.isfinite(result["geometry"][key])
    assert result["evaluation"]["num_samples"] == len(TEXTS)
    assert result["evaluation"]["gradients"] == "disabled"
    assert result["not_run"]["linear_probe"].startswith("not_run")

    _, export_info = load_student_compat(output / "student.pt")
    assert export_info["source_kind"] == "student_export"
    assert "student" not in PERSON2_LOADER_KEYS
    _, checkpoint_info = load_student_compat(output / "checkpoint.pt")
    assert checkpoint_info["source_kind"] == "training_checkpoint"
    compatible = person2_compatible_state(output / "student.pt")
    assert "student_encoder" in compatible
    assert compatible["model_config"] == export_info["model_config"]

    model, _ = load_student_compat(output / "student.pt")
    for parameter in model.parameters():
        parameter.data.zero_()
    tokenizer = sys.modules["src.data"].get_bert_tokenizer("stub-tokenizer")
    degenerate = evaluate_geometry(
        model, tokenizer, TEXTS, max_length=16, batch_size=2, device="cpu"
    )
    assert degenerate["geometry"]["is_degenerate"] is True
    assert degenerate["geometry"]["status"] == "constant_representations_collapsed"


def test_geometry_from_synthetic_features_is_person1_side_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_person2_stubs(monkeypatch)
    healthy = geometry_from_embeddings(torch.randn(6, 8, generator=torch.Generator()))
    assert healthy["is_degenerate"] is False
    assert math.isfinite(healthy["effective_rank_singular"])

    collapsed = geometry_from_embeddings(torch.ones(6, 8))
    assert collapsed["is_degenerate"] is True
    assert collapsed["status"] == "constant_representations_collapsed"


def test_bridge_fails_clearly_without_person2_evaluate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if importlib.util.find_spec("src.evaluate") is not None:
        pytest.skip(
            "Applies only to a person1-only tree; src/evaluate.py is present in this checkout"
        )
    install_person2_stubs(monkeypatch)
    monkeypatch.delitem(sys.modules, "src.evaluate", raising=False)
    with pytest.raises(RuntimeError, match="src/evaluate.py"):
        geometry_from_embeddings(torch.randn(4, 4))
