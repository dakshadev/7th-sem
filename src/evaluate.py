"""
src/evaluate.py

Person 2 owned module: representation evaluation, pooling, geometry diagnostics,
and downstream linear probing.
Adheres strictly to the evaluation contract defined in docs/INTERFACES.md:
- Uncorrupted input text to frozen student encoder in evaluation mode.
- Attention-mask-aware mean pooling excluding padding and special tokens.
- Centered effective rank (RankMe), per-dimension variance, and centered cosine distribution.
- Explicit degenerate status reporting for zero-norm or collapsed representations.
- Linear probe: scaler and classifier fitted strictly on train split.
"""

from typing import List, Dict, Optional, Tuple, Union, Any, Set
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def masked_mean_pooling(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    input_ids: torch.Tensor,
    special_token_ids: Union[Set[int], List[int]],
) -> torch.Tensor:
    """
    Computes attention-mask-aware mean pooling over valid sequence tokens,
    strictly excluding padding and all special tokens (CLS, SEP, PAD, etc.).

    Args:
        hidden_states: [B, L, D] tensor of representations from encoder.
        attention_mask: [B, L] bool tensor (True = valid token, False = padding).
        input_ids: [B, L] int64 tensor of uncorrupted token IDs.
        special_token_ids: Set or list of special token IDs to exclude from pooling.

    Returns:
        sentence_vectors: [B, D] tensor of pooled representation vectors.
    """
    device = hidden_states.device
    special_ids_tensor = torch.tensor(list(special_token_ids), device=device, dtype=input_ids.dtype)

    # Valid mask: within attention_mask AND not matching any special token ID
    is_special = torch.isin(input_ids, special_ids_tensor)
    valid_mask = attention_mask.bool() & ~is_special  # [B, L]

    # Expand for multiplication with [B, L, D]
    valid_mask_expanded = valid_mask.unsqueeze(-1).to(hidden_states.dtype)  # [B, L, 1]

    # Sum representations over valid tokens
    sum_hidden = (hidden_states * valid_mask_expanded).sum(dim=1)  # [B, D]

    # Count of valid tokens per example
    valid_counts = valid_mask.sum(dim=1, keepdim=True).to(hidden_states.dtype)  # [B, 1]

    # Safe division: if an example has 0 valid tokens, clamp denominator to 1.0
    # The numerator is 0, so the resulting vector is safely all zeros (guard empty examples)
    safe_counts = torch.clamp(valid_counts, min=1.0)
    pooled = sum_hidden / safe_counts

    return pooled


def compute_representation_geometry(
    embeddings: Union[torch.Tensor, np.ndarray],
    eps: float = 1e-12,
    dead_dim_threshold: float = 1e-6,
) -> Dict[str, Any]:
    """
    Computes rigorous representation geometry metrics for a sample matrix Z [N, D].

    Definitions:
    - N: sample count, D: embedding dimension.
    - Centering: Z_tilde = Z - mean(Z, axis=0).
    - Centered Effective Rank (RankMe):
      SVD on Z_tilde gives singular values s_k.
      Normalized p_k = s_k / sum(s_j).
      RankMe = exp(- sum p_k * ln(p_k)). Bounded in [1, min(N, D)].
    - Covariance Effective Rank:
      Eigenvalues lambda_k = s_k^2 / (N - 1).
      Normalized q_k = lambda_k / sum(lambda_j).
      CovRank = exp(- sum q_k * ln(q_k)).
    - Dimension Variance:
      Sample variance along samples for each of the D dimensions.
    - Centered Pairwise Cosine:
      Pairwise cosine similarities of normalized centered vectors.

    Degenerate Cases:
    - Returns is_degenerate=True with explicit status reason for:
      insufficient samples (N < 2), NaN/Inf, all-zero spectrum, or collapsed variance.
    """
    if isinstance(embeddings, torch.Tensor):
        Z = embeddings.detach().cpu().to(torch.float64).numpy()
    else:
        Z = np.asarray(embeddings, dtype=np.float64)

    # 1. Check finiteness
    if not np.isfinite(Z).all():
        return {
            "is_degenerate": True,
            "status": "nan_or_inf_detected",
            "num_samples": int(Z.shape[0]),
            "dimension": int(Z.shape[1]) if Z.ndim > 1 else 0,
        }

    N = Z.shape[0]
    if N < 2:
        return {
            "is_degenerate": True,
            "status": "insufficient_samples_N_less_than_2",
            "num_samples": N,
            "dimension": int(Z.shape[1]) if Z.ndim > 1 else 0,
        }

    D = Z.shape[1]

    # 2. Centering along sample axis
    mean_vec = np.mean(Z, axis=0, keepdims=True)
    Z_centered = Z - mean_vec

    # 3. Per-dimension sample variance (N - 1 in denominator)
    dim_variance = np.var(Z_centered, axis=0, ddof=1)
    mean_dim_var = float(np.mean(dim_variance))
    min_dim_var = float(np.min(dim_variance))
    max_dim_var = float(np.max(dim_variance))
    dead_dims = int(np.sum(dim_variance < dead_dim_threshold))
    dead_dim_frac = float(dead_dims / D)

    # Check for complete representation collapse (all vectors identical / zero variance)
    total_var = float(np.sum(dim_variance))
    if total_var < eps:
        return {
            "is_degenerate": True,
            "status": "constant_representations_collapsed",
            "num_samples": N,
            "dimension": D,
            "effective_rank_singular": 1.0,
            "effective_rank_covariance": 1.0,
            "mean_dimension_variance": mean_dim_var,
            "min_dimension_variance": min_dim_var,
            "max_dimension_variance": max_dim_var,
            "dead_dimension_count": dead_dims,
            "dead_dimension_fraction": dead_dim_frac,
        }

    # 4. SVD on centered matrix
    try:
        # Singular values sorted in descending order
        singular_values = np.linalg.svd(Z_centered, full_matrices=False, compute_uv=False)
    except np.linalg.LinAlgError:
        return {
            "is_degenerate": True,
            "status": "svd_failed_to_converge",
            "num_samples": N,
            "dimension": D,
        }

    sv_sum = float(np.sum(singular_values))
    if sv_sum < eps:
        return {
            "is_degenerate": True,
            "status": "all_zero_spectrum",
            "num_samples": N,
            "dimension": D,
            "effective_rank_singular": 1.0,
            "effective_rank_covariance": 1.0,
            "mean_dimension_variance": mean_dim_var,
            "min_dimension_variance": min_dim_var,
            "max_dimension_variance": max_dim_var,
            "dead_dimension_count": dead_dims,
            "dead_dimension_fraction": dead_dim_frac,
        }

    # RankMe: entropy of normalized singular values
    p = singular_values / sv_sum
    p_safe = np.clip(p, eps, 1.0)
    entropy_sv = -float(np.sum(p_safe * np.log(p_safe)))
    effective_rank_singular = float(np.exp(entropy_sv))

    # Covariance eigenvalue effective rank: lambda = s^2 / (N - 1)
    eigenvalues = (singular_values ** 2) / (N - 1)
    eig_sum = float(np.sum(eigenvalues))
    if eig_sum > eps:
        q = eigenvalues / eig_sum
        q_safe = np.clip(q, eps, 1.0)
        entropy_eig = -float(np.sum(q_safe * np.log(q_safe)))
        effective_rank_covariance = float(np.exp(entropy_eig))
    else:
        effective_rank_covariance = 1.0

    # 5. Centered Pairwise Cosine Similarities
    # Normalize centered vectors to unit norm
    norms = np.linalg.norm(Z_centered, axis=1, keepdims=True)
    zero_norm_mask = (norms.squeeze(-1) < eps)
    if np.all(zero_norm_mask):
        return {
            "is_degenerate": True,
            "status": "zero_norm_centered_vectors",
            "num_samples": N,
            "dimension": D,
            "effective_rank_singular": effective_rank_singular,
            "effective_rank_covariance": effective_rank_covariance,
            "mean_dimension_variance": mean_dim_var,
            "dead_dimension_fraction": dead_dim_frac,
        }

    safe_norms = np.where(norms < eps, 1.0, norms)
    Z_normed = Z_centered / safe_norms

    # Pairwise cosine matrix: [N, N]
    cosine_matrix = np.matmul(Z_normed, Z_normed.T)
    # Extract upper triangular off-diagonal elements
    triu_indices = np.triu_indices(N, k=1)
    pair_cosines = cosine_matrix[triu_indices]

    mean_cos = float(np.mean(pair_cosines))
    std_cos = float(np.std(pair_cosines))
    min_cos = float(np.min(pair_cosines))
    max_cos = float(np.max(pair_cosines))
    median_cos = float(np.median(pair_cosines))
    p10_cos = float(np.percentile(pair_cosines, 10))
    p90_cos = float(np.percentile(pair_cosines, 90))

    return {
        "is_degenerate": False,
        "status": "healthy",
        "num_samples": N,
        "dimension": D,
        "effective_rank_singular": effective_rank_singular,
        "effective_rank_covariance": effective_rank_covariance,
        "rank_ratio": float(effective_rank_singular / min(N, D)),
        "mean_dimension_variance": mean_dim_var,
        "min_dimension_variance": min_dim_var,
        "max_dimension_variance": max_dim_var,
        "dead_dimension_count": dead_dims,
        "dead_dimension_fraction": dead_dim_frac,
        "mean_pairwise_cosine": mean_cos,
        "std_pairwise_cosine": std_cos,
        "min_pairwise_cosine": min_cos,
        "max_pairwise_cosine": max_cos,
        "median_pairwise_cosine": median_cos,
        "p10_pairwise_cosine": p10_cos,
        "p90_pairwise_cosine": p90_cos,
    }


def freeze_model_for_evaluation(model: nn.Module) -> nn.Module:
    """
    Prepares student encoder for evaluation:
    - Sets model to eval mode
    - Disables gradients on all parameters
    """
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def read_checkpoint_payload(checkpoint: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Reads a checkpoint payload from a file path or dict.
    Extracts student weights, model configuration, and training step.
    Supports 'student', 'student_encoder', 'model', or 'encoder' keys.
    """
    if isinstance(checkpoint, str):
        state = torch.load(checkpoint, map_location="cpu")
    else:
        state = checkpoint

    if not isinstance(state, dict):
        raise ValueError(f"Expected dict or checkpoint path, got {type(state)}")

    # Extract student weights
    if "student" in state:
        weights = state["student"]
    elif "student_encoder" in state:
        weights = state["student_encoder"]
    elif "model" in state:
        weights = state["model"]
    elif "encoder" in state:
        weights = state["encoder"]
    else:
        weights = state

    # Read model configuration if present
    model_config = state.get("model_config") or state.get("config")
    step = state.get("step")

    return {
        "student_state_dict": weights,
        "model_config": model_config,
        "step": step,
        "raw_checkpoint": state,
    }


def load_student_checkpoint(
    checkpoint: Union[str, Dict[str, Any]],
    model: nn.Module,
    strict: bool = True,
    return_config: bool = False,
) -> Union[nn.Module, Tuple[nn.Module, Optional[Dict[str, Any]]]]:
    """
    Loads student weights from a checkpoint path or state dict.
    Accepts 'student', 'student_encoder', 'model', or 'encoder' keys,
    reads 'model_config' (or 'config'), attaches it to the model,
    freezes the model, and puts it in eval mode.

    Args:
        checkpoint: File path or dictionary containing checkpoint.
        model: Target PyTorch module to load weights into.
        strict: Whether to strictly enforce state_dict key matching.
        return_config: If True, returns tuple (frozen_model, model_config).
                       If False, returns frozen_model.
    """
    payload = read_checkpoint_payload(checkpoint)
    weights = payload["student_state_dict"]
    model_config = payload["model_config"]

    model.load_state_dict(weights, strict=strict)
    model.model_config = model_config
    model.config = model_config
    frozen_model = freeze_model_for_evaluation(model)

    if return_config:
        return frozen_model, model_config
    return frozen_model



def evaluate_frozen_linear_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    seed: int = 42,
    max_iter: int = 1000,
    C: float = 1.0,
) -> Dict[str, Any]:
    """
    Fits and evaluates a frozen linear probe (e.g. AG News).
    Invariant: Standard scaler and classifier are fit ONLY on train split.
    Test features are transformed using the train-fitted scaler.

    Returns:
        accuracy: Held-out test accuracy.
        macro_f1: Held-out test macro-F1 score.
        majority_baseline: Baseline accuracy achieved by majority class.
        train_samples: Number of training samples.
        test_samples: Number of evaluation samples.
        num_classes: Number of distinct classes.
    """
    assert len(train_features) == len(train_labels), "Train features and labels length mismatch"
    assert len(test_features) == len(test_labels), "Test features and labels length mismatch"
    assert len(train_features) > 0, "Empty train features"
    assert len(test_features) > 0, "Empty test features"

    # Fit scaler strictly on train split
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)

    # Compute majority class baseline on test set
    unique_classes, test_counts = np.unique(test_labels, return_counts=True)
    majority_baseline = float(np.max(test_counts) / len(test_labels))

    # Train linear classifier
    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=seed,
        solver="lbfgs",
    )
    clf.fit(train_scaled, train_labels)

    # Predict on held-out test split
    predictions = clf.predict(test_scaled)
    acc = float(accuracy_score(test_labels, predictions))
    f1 = float(f1_score(test_labels, predictions, average="macro"))

    return {
        "accuracy": acc,
        "macro_f1": f1,
        "majority_baseline": majority_baseline,
        "train_samples": int(len(train_features)),
        "test_samples": int(len(test_features)),
        "num_classes": int(len(unique_classes)),
    }
