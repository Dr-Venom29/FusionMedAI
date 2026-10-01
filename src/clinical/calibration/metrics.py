"""
Metrics for clinical probability calibration and risk reliability.
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)
from sklearn.linear_model import LogisticRegression


def compute_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute calibration curve: bin centers/means, empirical observed event rates, and bin counts.
    
    Args:
        y_true: Binary ground-truth labels (0 or 1).
        y_prob: Predicted event probabilities in [0, 1].
        n_bins: Number of probability bins.
        strategy: 'uniform' (equal-width) or 'quantile' (equal-frequency).
        
    Returns:
        prob_pred: Mean predicted probability in each non-empty bin.
        prob_true: Observed true positive rate in each non-empty bin.
        bin_counts: Number of samples in each non-empty bin.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.clip(np.asarray(y_prob).ravel(), 1e-12, 1.0 - 1e-12)

    if strategy == "quantile":
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.percentile(y_prob, quantiles * 100)
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0
    else:
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    prob_pred_list = []
    prob_true_list = []
    counts_list = []

    for i in range(n_bins):
        if i == n_bins - 1:
            idx = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        else:
            idx = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])

        if np.sum(idx) > 0:
            prob_pred_list.append(float(np.mean(y_prob[idx])))
            prob_true_list.append(float(np.mean(y_true[idx])))
            counts_list.append(int(np.sum(idx)))

    return np.array(prob_pred_list), np.array(prob_true_list), np.array(counts_list)


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    """
    prob_pred, prob_true, bin_counts = compute_calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy=strategy
    )
    if len(bin_counts) == 0:
        return 0.0
    total_samples = np.sum(bin_counts)
    abs_errors = np.abs(prob_true - prob_pred)
    ece = np.sum((bin_counts / total_samples) * abs_errors)
    return float(ece)


def compute_mce(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """
    Compute Maximum Calibration Error (MCE).
    """
    prob_pred, prob_true, bin_counts = compute_calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy=strategy
    )
    if len(bin_counts) == 0:
        return 0.0
    abs_errors = np.abs(prob_true - prob_pred)
    return float(np.max(abs_errors))


def compute_calibration_slope_intercept(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[float, float]:
    """
    Fit logistic calibration line: logit(y) ~ intercept + slope * logit(p).
    
    Target ideal:
        intercept = 0.0 (unbiased base rate)
        slope = 1.0 (correct risk spread)
    """
    y_true = np.asarray(y_true).ravel()
    eps = 1e-12
    y_prob_clipped = np.clip(np.asarray(y_prob).ravel(), eps, 1.0 - eps)
    logits = np.log(y_prob_clipped / (1.0 - y_prob_clipped)).reshape(-1, 1)

    try:
        # Logistic regression without regularization to evaluate true empirical slope/intercept
        clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        clf.fit(logits, y_true)
        slope = float(clf.coef_[0, 0])
        intercept = float(clf.intercept_[0])
    except Exception:
        # Fallback with minimal L2 regularization if unpenalized fails
        clf = LogisticRegression(C=1e5, solver="lbfgs", max_iter=1000)
        clf.fit(logits, y_true)
        slope = float(clf.coef_[0, 0])
        intercept = float(clf.intercept_[0])

    return intercept, slope


def evaluate_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute comprehensive discrimination and probability calibration evaluation suite.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.clip(np.asarray(y_prob).ravel(), 1e-12, 1.0 - 1e-12)

    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    loss = float(log_loss(y_true, y_prob))
    ece_uniform = compute_ece(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    ece_quantile = compute_ece(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    mce = compute_mce(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    intercept, slope = compute_calibration_slope_intercept(y_true, y_prob)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "log_loss": loss,
        "ece": ece_uniform,
        "ece_quantile": ece_quantile,
        "mce": mce,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }
