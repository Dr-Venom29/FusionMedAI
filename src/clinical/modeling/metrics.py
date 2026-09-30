"""Evaluation Metrics Interface for Clinical Tabular Models.

Computes:
- Discrimination: ROC-AUC, PR-AUC (Average Precision)
- Calibration: Brier Score, Log Loss, Calibration Curve
- Threshold-Dependent: Sensitivity, Specificity, PPV, NPV, F1 across [0.10, 0.20, 0.30, 0.40, 0.50]
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    model_name: str = "model",
) -> Dict[str, Any]:
    """Compute standard discrimination, calibration, and classification metrics at a given threshold."""
    # Ensure binary format
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    # Discrimination metrics
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))

    # Calibration metrics
    brier = float(brier_score_loss(y_true, y_prob))
    ll = float(log_loss(y_true, y_prob))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Threshold-dependent metrics
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    f1 = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0

    return {
        "model": model_name,
        "threshold": threshold,
        "n_samples": int(len(y_true)),
        "prevalence": float(np.mean(y_true)),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "log_loss": round(ll, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "ppv": round(ppv, 4),
        "npv": round(npv, 4),
        "f1": round(f1, 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def compute_threshold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: List[float] = [0.10, 0.20, 0.30, 0.40, 0.50],
    model_name: str = "model",
) -> List[Dict[str, Any]]:
    """Evaluate performance across an array of clinical decision thresholds."""
    results = []
    for th in thresholds:
        m = compute_classification_metrics(y_true, y_prob, threshold=th, model_name=model_name)
        results.append(m)
    return results


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    model_name: str = "model",
) -> Dict[str, Any]:
    """Compute calibration curve data and expected calibration error (ECE)."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    
    # Compute ECE
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_details = []
    for i in range(n_bins):
        idx = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            idx = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        bin_count = int(np.sum(idx))
        if bin_count > 0:
            obs = float(np.mean(y_true[idx]))
            pred = float(np.mean(y_prob[idx]))
            diff = abs(obs - pred)
            ece += (bin_count / len(y_true)) * diff
            bin_details.append({
                "bin": i + 1,
                "bin_lower": round(float(bin_edges[i]), 2),
                "bin_upper": round(float(bin_edges[i + 1]), 2),
                "count": bin_count,
                "observed_frequency": round(obs, 4),
                "mean_predicted_prob": round(pred, 4),
                "calibration_gap": round(diff, 4),
            })

    return {
        "model": model_name,
        "n_bins": n_bins,
        "expected_calibration_error": round(float(ece), 4),
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
        "log_loss": round(float(log_loss(y_true, y_prob)), 4),
        "bins": bin_details,
    }
