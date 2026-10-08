"""
src/fusion/calibration/calibration_metrics.py
Phase C11.11: Modality & Fusion Decision Calibration Metrics

Strict Methodological Boundary:
- Modality-level metrics (ECE, Brier, NLL, Slope, Intercept) are strictly computed on
  each constituent modality's legitimate single-task validation/test splits with true ground truth.
- Fusion-level risk (R_fusion) and DCRI are derived decision indices across unpaired cohorts.
  No synthetic multimodal ground truth is fabricated, and fusion-level ECE/Brier are explicitly disallowed.
"""

from typing import Dict, Any, List, Tuple, Sequence, Optional
import numpy as np
import math


def compute_modality_ece(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Computes Expected Calibration Error (ECE) for multi-class or binary predictions.
    
    Args:
        probabilities: [N, K] posterior class probabilities or [N] binary probabilities.
        labels: [N] true integer class labels in {0, ..., K-1}.
        n_bins: Number of equal-width confidence bins in [0, 1].
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    y_true = np.asarray(labels, dtype=np.int64).ravel()

    if probs.ndim == 1:
        # Binary case
        confidences = np.maximum(probs, 1.0 - probs)
        predictions = (probs >= 0.5).astype(np.int64)
    else:
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(predictions[in_bin] == y_true[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_count / n_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)


def compute_modality_brier(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Computes multi-class or binary Brier score."""
    probs = np.asarray(probabilities, dtype=np.float64)
    y_true = np.asarray(labels, dtype=np.int64).ravel()
    n_samples = len(y_true)

    if probs.ndim == 1:
        # Binary
        brier = np.mean((probs - y_true) ** 2)
    else:
        n_classes = probs.shape[1]
        y_one_hot = np.zeros((n_samples, n_classes), dtype=np.float64)
        y_one_hot[np.arange(n_samples), y_true] = 1.0
        brier = np.mean(np.sum((probs - y_one_hot) ** 2, axis=1))

    return float(brier)


def compute_modality_nll(
    probabilities: np.ndarray,
    labels: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Computes negative log-likelihood (cross-entropy)."""
    probs = np.clip(np.asarray(probabilities, dtype=np.float64), eps, 1.0)
    y_true = np.asarray(labels, dtype=np.int64).ravel()
    n_samples = len(y_true)

    if probs.ndim == 1:
        nll = -np.mean(y_true * np.log(probs) + (1.0 - y_true) * np.log(1.0 - probs))
    else:
        log_probs = np.log(probs)
        nll = -np.mean(log_probs[np.arange(n_samples), y_true])

    return float(nll)


def compute_shannon_entropy(probabilities: Sequence[float], eps: float = 1e-12) -> float:
    """Computes normalized Shannon entropy H(p) = -sum p_i log(p_i) / log(K)."""
    p = np.clip(np.asarray(probabilities, dtype=np.float64), eps, 1.0)
    p = p / np.sum(p)
    k = len(p)
    if k <= 1:
        return 0.0
    h = -np.sum(p * np.log(p))
    max_h = np.log(k)
    return float(np.clip(h / max_h, 0.0, 1.0))


def compute_routing_entropy(weights: Dict[str, float], eps: float = 1e-12) -> float:
    """
    Computes Shannon entropy of active routing weights:
        H(w) = -sum w_i * log2(w_i) / log2(M_active)
    """
    active_weights = [v for v in weights.values() if v > 1e-7]
    if len(active_weights) <= 1:
        return 0.0
    w = np.asarray(active_weights, dtype=np.float64)
    w = w / np.sum(w)
    entropy = -np.sum(w * np.log2(np.clip(w, eps, 1.0)))
    max_entropy = np.log2(len(active_weights))
    return float(np.clip(entropy / max_entropy, 0.0, 1.0))


def compute_conflict_index(risks: Dict[str, float]) -> float:
    """
    Computes the maximum pairwise risk discrepancy among available modalities:
        Delta_R = max_{i, j} |r_i - r_j|
    """
    r_vals = list(risks.values())
    if len(r_vals) < 2:
        return 0.0
    diffs = [abs(r_vals[i] - r_vals[j]) for i in range(len(r_vals)) for j in range(i + 1, len(r_vals))]
    return float(max(diffs)) if diffs else 0.0
