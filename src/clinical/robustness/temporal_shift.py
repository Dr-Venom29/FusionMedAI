"""
Temporal Shift & Longitudinal Stability Analysis (Phase C9).
Evaluates chronological drift across the 10-year study window (1999-2008) using encounter sequence proxies.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from src.clinical.robustness.metrics import evaluate_full_robustness_profile


def evaluate_temporal_shift(
    df_test_raw: pd.DataFrame,
    y_test: np.ndarray,
    cal_prob_test: np.ndarray,
    uncertainty_test: np.ndarray,
    threshold: float = 0.20,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluate temporal stability by partitioning the locked test set across chronological encounter progression.
    """
    # Nominal reference
    nom_metrics = evaluate_full_robustness_profile(
        y_true=y_test,
        y_prob=cal_prob_test,
        uncertainty=uncertainty_test,
        threshold=threshold,
    )

    # Encounter IDs provide chronological proxy ordering
    enc_ids = df_test_raw["encounter_id"].values
    median_enc = np.median(enc_ids)
    q25_enc = np.percentile(enc_ids, 25)
    q75_enc = np.percentile(enc_ids, 75)

    temporal_slices = {
        "Nominal_Test_Reference": ("Full Period", "Nominal 1999-2008 Test Cohort", np.ones(len(y_test), dtype=bool)),
        "Temporal_Early_Era": ("Chronological Half", "Early Era (~1999-2003, Lower 50% Encounter ID)", enc_ids < median_enc),
        "Temporal_Late_Era": ("Chronological Half", "Late Era (~2004-2008, Upper 50% Encounter ID)", enc_ids >= median_enc),
        "Temporal_Q1_Earliest": ("Chronological Quartile", "Quartile 1: Earliest Admissions (0-25%)", enc_ids < q25_enc),
        "Temporal_Q2_EarlyMid": ("Chronological Quartile", "Quartile 2: Early-Mid Admissions (25-50%)", (enc_ids >= q25_enc) & (enc_ids < median_enc)),
        "Temporal_Q3_LateMid": ("Chronological Quartile", "Quartile 3: Late-Mid Admissions (50-75%)", (enc_ids >= median_enc) & (enc_ids < q75_enc)),
        "Temporal_Q4_Latest": ("Chronological Quartile", "Quartile 4: Latest Admissions (75-100%)", enc_ids >= q75_enc),
    }

    rows = []
    detailed = {}

    for k, (cat, label, mask) in temporal_slices.items():
        n_s = int(np.sum(mask))
        y_s = y_test[mask]
        p_s = cal_prob_test[mask]
        u_s = uncertainty_test[mask]

        metrics = evaluate_full_robustness_profile(
            y_true=y_s,
            y_prob=p_s,
            uncertainty=u_s,
            threshold=threshold,
        )

        detailed[k] = metrics

        rows.append({
            "temporal_id": k,
            "category": cat,
            "period_description": label,
            "n_samples": n_s,
            "cohort_share_pct": (n_s / len(y_test)) * 100.0,
            "prevalence": metrics["prevalence"],
            "roc_auc": metrics["roc_auc"],
            "delta_roc_auc": metrics["roc_auc"] - nom_metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "delta_pr_auc": metrics["pr_auc"] - nom_metrics["pr_auc"],
            "brier_score": metrics["brier_score"],
            "ece": metrics["ece"],
            "calibration_slope": metrics["calibration_slope"],
            "mean_uncertainty": metrics["mean_uncertainty"],
            "median_uncertainty": metrics["median_uncertainty"],
            "p95_uncertainty": metrics["p95_uncertainty"],
            "error_rate": metrics["error_rate"],
            "delta_error_rate": metrics["error_rate"] - nom_metrics["error_rate"],
            "sensitivity": metrics["sensitivity"],
            "specificity": metrics["specificity"],
            "ppv": metrics["ppv"],
            "npv": metrics["npv"],
            "error_detection_auroc": metrics["error_detection_auroc"],
            "aurc": metrics["aurc"],
            "error_at_80pct_coverage": metrics["error_at_80pct_coverage"],
        })

    return pd.DataFrame(rows), detailed
