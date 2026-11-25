"""Deterministic, network-free tests for the probe redesign."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from scripts import build_evidence_tables, run_probe
from src.config import ModelConfig
from src.model import SmallEncoder

TINY = ModelConfig(
    vocab_size=16,
    max_length=16,
    width=16,
    layers=2,
    heads=4,
    ffn_width=32,
    dropout=0.0,
    pad_token_id=0,
    special_token_ids=(0, 1, 2, 3),
)


class TinyTokenizer:
    all_special_ids = [0, 1, 2, 3]

    def __call__(
        self,
        texts: str | list[str],
        max_length: int = 16,
        truncation: bool = True,
        padding: str = "max_length",
        return_attention_mask: bool = True,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        del truncation, padding, return_attention_mask, return_tensors
        items = [texts] if isinstance(texts, str) else list(texts)
        rows = []
        masks = []
        for text in items:
            tokens = [1] + [5 + (ord(char) % 8) for char in text[: max_length - 2]] + [2]
            tokens = tokens[:max_length]
            padding_count = max_length - len(tokens)
            rows.append(tokens + [0] * padding_count)
            masks.append([1] * len(tokens) + [0] * padding_count)
        return {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


def test_stratified_indices_are_balanced_deterministic_and_distinct() -> None:
    labels = [0] * 50 + [1] * 30 + [2] * 20
    first = run_probe.stratified_indices(labels, 30, seed=7)
    second = run_probe.stratified_indices(labels, 30, seed=7)
    assert first == second
    assert len(first) == 30
    assert len(set(first)) == 30
    counts = {label: sum(labels[index] == label for index in first) for label in (0, 1, 2)}
    assert counts == {0: 10, 1: 10, 2: 10}

    different = run_probe.stratified_indices(labels, 30, seed=8)
    assert different != first


def test_stratified_indices_fill_shortfall_from_small_classes() -> None:
    labels = [0] * 100 + [1] * 2
    chosen = run_probe.stratified_indices(labels, 20, seed=3)
    assert len(chosen) == 20
    assert len(set(chosen)) == 20
    assert sum(labels[index] == 1 for index in chosen) == 2
    with pytest.raises(ValueError, match="positive"):
        run_probe.stratified_indices(labels, 0, seed=1)


def test_summarise_reports_a_band() -> None:
    summary = run_probe.summarise([0.5, 0.6, 0.7])
    assert summary["mean"] == pytest.approx(0.6)
    assert summary["min"] == 0.5
    assert summary["max"] == 0.7
    assert summary["sd"] == pytest.approx(0.08164965809277261)
    with pytest.raises(ValueError, match="empty"):
        run_probe.summarise([])


def test_extract_feature_matrix_returns_every_layer_and_pooling() -> None:
    model = SmallEncoder(TINY).eval()
    features = run_probe.extract_feature_matrix(
        model,
        TinyTokenizer(),
        ["the quick fox", "a slow dog", "another line"],
        max_length=16,
        batch_size=2,
        device=torch.device("cpu"),
        special_token_ids=TINY.special_token_ids,
        layer_grid=True,
    )
    assert set(features) == {
        "layer_1_mean",
        "layer_1_cls",
        "layer_2_mean",
        "layer_2_cls",
    }
    for value in features.values():
        assert value.shape == (3, TINY.width)

    means_only = run_probe.extract_feature_matrix(
        model,
        TinyTokenizer(),
        ["the quick fox"],
        max_length=16,
        batch_size=1,
        device=torch.device("cpu"),
        special_token_ids=TINY.special_token_ids,
        layer_grid=False,
    )
    assert set(means_only) == {"layer_1_mean", "layer_2_mean"}


def test_probe_student_reports_final_layer_mean_as_the_headline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.data as data_module

    monkeypatch.setattr(data_module, "get_bert_tokenizer", lambda _name="tiny": TinyTokenizer())
    export = tmp_path / "student.pt"
    torch.save(
        {
            "schema_version": 1,
            "model_config": asdict(TINY),
            "student": SmallEncoder(TINY).state_dict(),
        },
        export,
    )
    train = (
        [f"football goal {index}" for index in range(8)],
        [0, 1] * 4,
        {"used": 8},
    )
    test = (["football goal"] * 4, [0, 1] * 2, {"used": 4})
    result = run_probe.probe_student(
        export,
        train=train,
        test=test,
        tokenizer_name="tiny",
        max_length=16,
        batch_size=2,
        seed=0,
        device=torch.device("cpu"),
    )
    assert result["layers"] == 2
    assert set(result["layer_pooling"]) == {
        "layer_1_mean",
        "layer_1_cls",
        "layer_2_mean",
        "layer_2_cls",
    }
    assert result["accuracy"] == result["layer_pooling"]["layer_2_mean"]["accuracy"]
    assert result["feature_shape"] == [8, TINY.width]


def test_random_baseline_reports_a_seed_band(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.data as data_module

    monkeypatch.setattr(data_module, "get_bert_tokenizer", lambda _name="tiny": TinyTokenizer())
    train = (["football goal"] * 8, [0, 1] * 4, {"used": 8})
    test = (["football goal"] * 4, [0, 1] * 2, {"used": 4})
    result = run_probe.random_baseline(
        train=train,
        test=test,
        model_config=asdict(TINY),
        seeds=[1, 2, 3],
        tokenizer_name="tiny",
        max_length=16,
        batch_size=2,
        device=torch.device("cpu"),
    )
    assert result["seeds"] == [1, 2, 3]
    assert len(result["accuracies"]) == 3
    assert result["accuracy"] == pytest.approx(sum(result["accuracies"]) / 3)
    assert result["accuracy_min"] <= result["accuracy"] <= result["accuracy_max"]
    assert result["accuracy_sd"] >= 0.0
    assert result["feature_shape"] == [8, TINY.width]


def test_lexical_baseline_separates_a_toy_task() -> None:
    train = (
        [f"football goal match {index}" for index in range(10)]
        + [f"stocks market shares {index}" for index in range(10)],
        [1] * 10 + [2] * 10,
        {"used": 20},
    )
    test = (
        ["football goal match"] * 5 + ["stocks market shares"] * 5,
        [1] * 5 + [2] * 5,
        {"used": 10},
    )
    result = run_probe.lexical_baseline(
        train=train, test=test, seed=0, ngram_range=(1, 1), min_df=1
    )
    assert result["accuracy"] >= 0.9
    assert result["num_features"] > 0
    assert result["fit_on"].startswith("probe-train only")


def test_evidence_tables_summarise_runs_and_params() -> None:
    runs = [
        {
            "label": "C",
            "budget_tokens": 1000,
            "seconds": 10.0,
            "nonpadding_tokens": 1000,
            "probe_accuracy": 0.5,
        },
        {
            "label": "C",
            "budget_tokens": 1000,
            "seconds": 20.0,
            "nonpadding_tokens": 2000,
            "probe_accuracy": 0.6,
        },
    ]
    rows = build_evidence_tables.summarise_runs(runs)
    assert len(rows) == 1
    assert rows[0]["runs"] == 2
    assert rows[0]["seconds"] == pytest.approx(15.0)
    assert rows[0]["tokens_per_second"] == pytest.approx(100.0)
    assert rows[0]["accuracy"] == pytest.approx(0.55)

    markdown = build_evidence_tables.render(runs, ["C"])
    assert "11,005,952" in markdown
    assert "15.0" in markdown
    assert "0.5500" in markdown
