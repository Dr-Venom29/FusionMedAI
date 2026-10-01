"""
Comprehensive metrics suite for Clinical Phase C9: Robustness, Fairness & Distribution Shift.
Computes Discrimination, Calibration Reliability, Classification at Operating Threshold,
and Uncertainty / Selective Prediction Performance.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.linear_model import LogisticRegression


def compute_calibration_slope_intercept(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[float, float]:
    """Fit logistic calibration line: logit(y) ~ intercept + slope * logit(p)."""
    y_true = np.asarray(y_true).ravel()
    eps = 1e-12
    y_prob_clipped = np.clip(np.asarray(y_prob).ravel(), eps, 1.0 - eps)
    logits = np.log(y_prob_clipped / (1.0 - y_prob_clipped)).reshape(-1, 1)

    try:
        clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        clf.fit(logits, y_true)
        slope = float(clf.coef_[0, 0])
        intercept = float(clf.intercept_[0])
    except Exception:
        clf = LogisticRegression(C=1e5, solver="lbfgs", max_iter=1000)
        clf.fit(logits, y_true)
        slope = float(clf.coef_[0, 0])
        intercept = float(clf.intercept_[0])

    return intercept, slope


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """Compute Expected Calibration Error (ECE)."""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.clip(np.asarray(y_prob).ravel(), 1e-12, 1.0 - 1e-12)

    if strategy == "quantile":
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.percentile(y_prob, quantiles * 100)
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0
    else:
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    bin_counts = []
    abs_errors = []

    for i in range(n_bins):
        if i == n_bins - 1:
            idx = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        else:
            idx = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])

        count = int(np.sum(idx))
        if count > 0:
            p_mean = float(np.mean(y_prob[idx]))
            y_mean = float(np.mean(y_true[idx]))
            bin_counts.append(count)
            abs_errors.append(abs(y_mean - p_mean))

    if not bin_counts:
        return 0.0

    total = np.sum(bin_counts)
    return float(np.sum((np.array(bin_counts) / total) * np.array(abs_errors)))


def compute_selective_prediction_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    uncertainty: np.ndarray,
    threshold: float = 0.20,
) -> Dict[str, float]:
    """Compute risk-coverage curve, AURC, E-AURC, and error at 80% coverage."""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    uncertainty = np.asarray(uncertainty).ravel()

    y_pred = (y_prob >= threshold).astype(int)
    errors = (y_pred != y_true).astype(int)
    n_samples = len(y_true)

    # Sort by uncertainty ascending
    order = np.argsort(uncertainty)
    errors_sorted = errors[order]

    # Evaluate at 100 coverage steps [0.01, 1.00]
    coverages = np.linspace(0.01, 1.00, 100)
    risks = []

    for cov in coverages:
        k = max(1, int(np.ceil(cov * n_samples)))
        retained_errors = errors_sorted[:k]
        risk = float(np.mean(retained_errors))
        risks.append(risk)

    risks = np.array(risks)
    aurc = float(np.trapz(risks, coverages))

    # Oracle selective predictor
    oracle_sorted_errors = np.sort(errors)  # all 0s first, then 1s
    oracle_risks = []
    for cov in coverages:
        k = max(1, int(np.ceil(cov * n_samples)))
        oracle_risks.append(float(np.mean(oracle_sorted_errors[:k])))
    aurc_oracle = float(np.trapz(oracle_risks, coverages))
    e_aurc = max(0.0, aurc - aurc_oracle)

    # Error at 80% coverage
    k_80 = max(1, int(np.ceil(0.80 * n_samples)))
    error_80 = float(np.mean(errors_sorted[:k_80]))

    return {
        "aurc": aurc,
        "aurc_oracle": aurc_oracle,
        "e_aurc": e_aurc,
        "error_at_80pct_coverage": error_80,
        "baseline_error_rate": float(np.mean(errors)),
    }


def evaluate_full_robustness_profile(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    uncertainty: np.ndarray,
    threshold: float = 0.20,
) -> Dict[str, Any]:
    """
    Compute comprehensive evaluation suite for a specific evaluation cohort / shift scenario.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.clip(np.asarray(y_prob).ravel(), 1e-12, 1.0 - 1e-12)
    uncertainty = np.asarray(uncertainty).ravel()
    n_samples = len(y_true)

    # 1. Discrimination
    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))
    else:
        roc_auc = 0.5
        pr_auc = float(np.mean(y_true))

    # 2. Probability Reliability / Calibration
    brier = float(brier_score_loss(y_true, y_prob))
    try:
        loss = float(log_loss(y_true, y_prob))
    except Exception:
        loss = 0.0

    ece = compute_ece(y_true, y_prob, n_bins=10, strategy="uniform")
    ece_quantile = compute_ece(y_true, y_prob, n_bins=10, strategy="quantile")
    intercept, slope = compute_calibration_slope_intercept(y_true, y_prob)

    # 3. Classification at Operating Threshold
    y_pred = (y_prob >= threshold).astype(int)
    errors = (y_pred != y_true).astype(int)
    error_rate = float(np.mean(errors))

    if len(np.unique(y_true)) > 1:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
    else:
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        sensitivity = specificity = ppv = npv = f1 = 0.0

    # 4. Uncertainty & Error Detection
    if len(np.unique(errors)) > 1:
        err_auroc = float(roc_auc_score(errors, uncertainty))
        err_auprc = float(average_precision_score(errors, uncertainty))
    else:
        err_auroc = 0.5
        err_auprc = float(np.mean(errors))

    sel_metrics = compute_selective_prediction_metrics(
        y_true, y_prob, uncertainty, threshold=threshold
    )

    unc_correct = uncertainty[errors == 0]
    unc_incorrect = uncertainty[errors == 1]

    return {
        "n_samples": n_samples,
        "prevalence": float(np.mean(y_true)),
        "mean_prob": float(np.mean(y_prob)),
        # Discrimination
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        # Calibration
        "brier_score": brier,
        "log_loss": loss,
        "ece": ece,
        "ece_quantile": ece_quantile,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        # Classification
        "threshold": threshold,
        "error_rate": error_rate,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        # Uncertainty
        "mean_uncertainty": float(np.mean(uncertainty)),
        "median_uncertainty": float(np.median(uncertainty)),
        "std_uncertainty": float(np.std(uncertainty)),
        "p25_uncertainty": float(np.percentile(uncertainty, 25)),
        "p75_uncertainty": float(np.percentile(uncertainty, 75)),
        "p90_uncertainty": float(np.percentile(uncertainty, 90)),
        "p95_uncertainty": float(np.percentile(uncertainty, 95)),
        "mean_unc_correct": float(np.mean(unc_correct)) if len(unc_correct) > 0 else 0.0,
        "mean_unc_incorrect": float(np.mean(unc_incorrect)) if len(unc_incorrect) > 0 else 0.0,
        "error_detection_auroc": err_auroc,
        "error_detection_auprc": err_auprc,
        "aurc": sel_metrics["aurc"],
        "e_aurc": sel_metrics["e_aurc"],
        "error_at_80pct_coverage": sel_metrics["error_at_80pct_coverage"],
    }
