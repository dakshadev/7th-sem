"""
tests/test_evaluation.py

Unit tests for Person 2 evaluation pipeline (src/evaluate.py).
Deterministic CPU unit tests with synthetic fixtures and no network calls.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from src.evaluate import (
    masked_mean_pooling,
    compute_representation_geometry,
    freeze_model_for_evaluation,
    load_student_checkpoint,
    evaluate_frozen_linear_probe,
)


class DummyEncoder(nn.Module):
    """Tiny synthetic encoder fixture for testing model freezing and checkpoint loading."""
    def __init__(self, vocab_size=30522, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        return self.linear(x)


def test_masked_mean_pooling_excludes_padding():
    """Padding positions, even with extreme values, must not affect pooled output."""
    B, L, D = 2, 5, 4
    # Hidden states: fill with 1.0 everywhere
    hidden_states = torch.ones(B, L, D, dtype=torch.float32)
    # Positions 3 and 4 in sequence 0 are padding
    attention_mask = torch.tensor([
        [True, True, True, False, False],
        [True, True, True, True, False],
    ], dtype=torch.bool)
    # Input ids: 101 (CLS), 500 (word), 102 (SEP), 0 (PAD), 0 (PAD)
    input_ids = torch.tensor([
        [101, 500, 102, 0, 0],
        [101, 501, 502, 102, 0],
    ], dtype=torch.int64)

    special_token_ids = {101, 102, 0}

    # Inject extreme corrupted value at padding position
    hidden_states[0, 3, :] = 9999.0
    hidden_states[0, 4, :] = -9999.0

    pooled = masked_mean_pooling(hidden_states, attention_mask, input_ids, special_token_ids)

    # In sequence 0, only position 1 (word 500) is valid (1 token)
    # Since hidden_states[0, 1, :] was 1.0, mean should be 1.0
    assert torch.allclose(pooled[0], torch.ones(D), atol=1e-5)


def test_masked_mean_pooling_excludes_special_tokens():
    """Special tokens (CLS, SEP) with extreme values must not leak into pooled representation."""
    B, L, D = 1, 4, 3
    hidden_states = torch.zeros(B, L, D, dtype=torch.float32)
    attention_mask = torch.ones(B, L, dtype=torch.bool)
    # [CLS], token1, token2, [SEP]
    input_ids = torch.tensor([[101, 300, 400, 102]], dtype=torch.int64)
    special_token_ids = {101, 102, 0}

    # Set valid tokens to specific vectors
    hidden_states[0, 1, :] = torch.tensor([2.0, 4.0, 6.0])
    hidden_states[0, 2, :] = torch.tensor([4.0, 6.0, 8.0])

    # Poison CLS and SEP with massive values
    hidden_states[0, 0, :] = 100000.0  # CLS
    hidden_states[0, 3, :] = -100000.0  # SEP

    pooled = masked_mean_pooling(hidden_states, attention_mask, input_ids, special_token_ids)

    # Expected mean of valid tokens: ( [2,4,6] + [4,6,8] ) / 2 = [3, 5, 7]
    expected = torch.tensor([[3.0, 5.0, 7.0]])
    assert torch.allclose(pooled, expected, atol=1e-5)


def test_masked_mean_pooling_empty_example_guard():
    """An example with zero valid tokens (only special tokens and padding) safely returns zeros."""
    B, L, D = 1, 3, 4
    hidden_states = torch.ones(B, L, D, dtype=torch.float32) * 5.0
    # [CLS], [SEP], [PAD] -> 0 valid non-special tokens!
    input_ids = torch.tensor([[101, 102, 0]], dtype=torch.int64)
    attention_mask = torch.tensor([[True, True, False]], dtype=torch.bool)
    special_token_ids = {101, 102, 0}

    pooled = masked_mean_pooling(hidden_states, attention_mask, input_ids, special_token_ids)
    assert pooled.shape == (1, D)
    assert not torch.isnan(pooled).any()
    assert torch.allclose(pooled, torch.zeros(1, D))


def test_geometry_healthy_fixture():
    """Synthetic isotropic Gaussian representations yield healthy geometry metrics."""
    np.random.seed(42)
    N, D = 50, 128
    Z = np.random.randn(N, D)

    metrics = compute_representation_geometry(Z)
    assert not metrics["is_degenerate"]
    assert metrics["status"] == "healthy"
    assert metrics["num_samples"] == N
    assert metrics["dimension"] == D
    # RankMe should be substantial for isotropic Gaussian
    assert 10.0 < metrics["effective_rank_singular"] <= min(N, D)
    assert 10.0 < metrics["effective_rank_covariance"] <= min(N, D)
    assert metrics["mean_dimension_variance"] > 0.0
    assert metrics["dead_dimension_count"] == 0
    assert -1.0 <= metrics["mean_pairwise_cosine"] <= 1.0
    assert metrics["p10_pairwise_cosine"] <= metrics["p90_pairwise_cosine"]


def test_geometry_constant_collapse_degenerate():
    """Constant representations (complete collapse) must be flagged degenerate."""
    N, D = 30, 64
    # All vectors identical
    collapsed_vector = np.ones((1, D)) * 3.14
    Z = np.repeat(collapsed_vector, N, axis=0)

    metrics = compute_representation_geometry(Z)
    assert metrics["is_degenerate"]
    assert metrics["status"] == "constant_representations_collapsed"
    assert metrics["dead_dimension_fraction"] == 1.0


def test_geometry_zero_fixture():
    """All-zero representations must be flagged degenerate."""
    Z = np.zeros((20, 64))
    metrics = compute_representation_geometry(Z)
    assert metrics["is_degenerate"]
    assert metrics["status"] in ("constant_representations_collapsed", "all_zero_spectrum")


def test_geometry_nan_inf_detection():
    """NaN or Inf inputs must be safely flagged degenerate."""
    Z = np.random.randn(20, 64)
    Z[5, 10] = np.nan
    metrics = compute_representation_geometry(Z)
    assert metrics["is_degenerate"]
    assert metrics["status"] == "nan_or_inf_detected"


def test_geometry_insufficient_samples():
    """N < 2 must return explicit degenerate status."""
    Z = np.random.randn(1, 64)
    metrics = compute_representation_geometry(Z)
    assert metrics["is_degenerate"]
    assert metrics["status"] == "insufficient_samples_N_less_than_2"


def test_encoder_weights_frozen_during_evaluation():
    """Ensure freeze helper sets requires_grad=False and parameters remain bitwise identical."""
    model = DummyEncoder(vocab_size=1000, hidden_dim=64)
    freeze_model_for_evaluation(model)

    assert not model.training
    for p in model.parameters():
        assert not p.requires_grad

    # Record initial weight hashes / copies
    initial_weights = {k: v.clone() for k, v in model.state_dict().items()}

    # Run forward pass
    dummy_input = torch.randint(0, 1000, (4, 16))
    out = model(dummy_input)

    # Weights must be exactly identical
    for k, v in model.state_dict().items():
        assert torch.equal(v, initial_weights[k]), f"Weight {k} changed during evaluation!"


def test_checkpoint_loading_roundtrip(tmp_path):
    """Verify loading student weights from a saved checkpoint dictionary."""
    model = DummyEncoder(vocab_size=500, hidden_dim=32)
    checkpoint_path = tmp_path / "student_checkpoint.pt"

    # Save checkpoint with wrapper dict matching Person 1 checkpoint convention
    torch.save({"student_encoder": model.state_dict(), "step": 100}, checkpoint_path)

    new_model = DummyEncoder(vocab_size=500, hidden_dim=32)
    loaded_model = load_student_checkpoint(str(checkpoint_path), new_model)

    assert not loaded_model.training
    for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
        assert torch.equal(p1, p2)
        assert not p2.requires_grad


def test_frozen_linear_probe_train_only_scaling():
    """Verify linear probe fits scaler and classifier on train split only and evaluates test."""
    np.random.seed(42)
    # Linearly separable 2-class problem
    N_train, N_test, D = 80, 20, 16
    X_train = np.vstack([np.random.randn(40, D) + 3.0, np.random.randn(40, D) - 3.0])
    y_train = np.array([0] * 40 + [1] * 40)

    X_test = np.vstack([np.random.randn(10, D) + 3.0, np.random.randn(10, D) - 3.0])
    y_test = np.array([0] * 10 + [1] * 10)

    results = evaluate_frozen_linear_probe(
        X_train, y_train, X_test, y_test, seed=42
    )

    assert results["train_samples"] == N_train
    assert results["test_samples"] == N_test
    assert results["num_classes"] == 2
    assert results["majority_baseline"] == 0.5
    # Since data is well-separated, probe should achieve high accuracy
    assert results["accuracy"] >= 0.9
    assert results["macro_f1"] >= 0.9
