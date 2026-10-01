"""
Decision Curve Analysis (DCA) for clinical net benefit evaluation.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd


def compute_net_benefit(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    """
    Compute Net Benefit across a range of clinical decision thresholds:
        Net Benefit = (TP / N) - (FP / N) * (threshold / (1 - threshold))
    
    Also computes:
        Treat-All Net Benefit = Prevalence - (1 - Prevalence) * (threshold / (1 - threshold))
        Treat-None Net Benefit = 0.0
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    n_total = len(y_true)
    n_pos = int(np.sum(y_true == 1))
    prevalence = n_pos / n_total

    results = []

    for th in thresholds:
        if th <= 0.0 or th >= 1.0:
            continue
        weight = th / (1.0 - th)

        # Model decisions
        y_pred = (y_prob >= th).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))

        net_benefit_model = (tp / n_total) - (fp / n_total) * weight

        # Treat-All benchmark
        net_benefit_all = prevalence - (1.0 - prevalence) * weight

        # Treat-None benchmark
        net_benefit_none = 0.0

        results.append({
            "threshold": float(th),
            "net_benefit_model": float(net_benefit_model),
            "net_benefit_all": float(net_benefit_all),
            "net_benefit_none": float(net_benefit_none),
            "tp": int(tp),
            "fp": int(fp),
            "flagged_count": int(tp + fp),
            "flagged_pct": float((tp + fp) / n_total * 100.0),
        })

    return pd.DataFrame(results)


def compute_operating_threshold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: List[float],
) -> pd.DataFrame:
    """
    Evaluate sensitivity, specificity, PPV, NPV, and F1 across candidate decision thresholds.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    n_total = len(y_true)

    rows = []
    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

        rows.append({
            "threshold": float(th),
            "sensitivity": float(sens),
            "specificity": float(spec),
            "ppv": float(ppv),
            "npv": float(npv),
            "f1": float(f1),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "flagged_count": tp + fp,
            "flagged_pct": (tp + fp) / n_total * 100.0,
        })

    return pd.DataFrame(rows)
