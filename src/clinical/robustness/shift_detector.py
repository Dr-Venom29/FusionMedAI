"""
Uncertainty as Shift Detector & Epistemic Failure Regime Profiling (Phase C9).
Quantifies uncertainty response to distribution shift and identifies silent high-confidence failure modes.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd


def profile_epistemic_failure_regimes(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    uncertainty: np.ndarray,
    threshold: float = 0.20,
    uncertainty_cutoff_pct: float = 75.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Profile encounters into four operational epistemic quadrants:
    Q1: Confident Correct (Low Uncertainty, Correct Classification)
    Q2: Cautious Correct (High Uncertainty, Correct Classification)
    Q3: Alerted Error (High Uncertainty, Misclassification - Desirable Warning!)
    Q4: High-Confidence Silent Failure (Low Uncertainty, Misclassification - Dangerous Failure Mode!)
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    uncertainty = np.asarray(uncertainty).ravel()

    y_pred = (y_prob >= threshold).astype(int)
    errors = (y_pred != y_true).astype(int)

    u_thresh = float(np.percentile(uncertainty, uncertainty_cutoff_pct))

    is_high_u = uncertainty >= u_thresh
    is_low_u = ~is_high_u

    is_error = errors == 1
    is_correct = errors == 0

    q1_mask = is_low_u & is_correct
    q2_mask = is_high_u & is_correct
    q3_mask = is_high_u & is_error
    q4_mask = is_low_u & is_error

    n_total = len(y_true)

    regimes = [
        {
            "regime_name": "Q1: Confident Correct",
            "clinical_status": "Optimal CDS Operation",
            "encounter_count": int(np.sum(q1_mask)),
            "cohort_share_pct": (np.sum(q1_mask) / n_total) * 100.0,
            "mean_prob": float(np.mean(y_prob[q1_mask])) if np.sum(q1_mask) > 0 else 0.0,
            "mean_uncertainty": float(np.mean(uncertainty[q1_mask])) if np.sum(q1_mask) > 0 else 0.0,
            "observed_prevalence": float(np.mean(y_true[q1_mask])) if np.sum(q1_mask) > 0 else 0.0,
            "description": "Model is confident and correct. Low-intensity automated triage is appropriate.",
        },
        {
            "regime_name": "Q2: Cautious Correct",
            "clinical_status": "Complex Successful Prediction",
            "encounter_count": int(np.sum(q2_mask)),
            "cohort_share_pct": (np.sum(q2_mask) / n_total) * 100.0,
            "mean_prob": float(np.mean(y_prob[q2_mask])) if np.sum(q2_mask) > 0 else 0.0,
            "mean_uncertainty": float(np.mean(uncertainty[q2_mask])) if np.sum(q2_mask) > 0 else 0.0,
            "observed_prevalence": float(np.mean(y_true[q2_mask])) if np.sum(q2_mask) > 0 else 0.0,
            "description": "Correct prediction despite elevated parameter ambiguity. Robust performance under complexity.",
        },
        {
            "regime_name": "Q3: Alerted Error (Warned Failure)",
            "clinical_status": "Actionable Alert Failure",
            "encounter_count": int(np.sum(q3_mask)),
            "cohort_share_pct": (np.sum(q3_mask) / n_total) * 100.0,
            "mean_prob": float(np.mean(y_prob[q3_mask])) if np.sum(q3_mask) > 0 else 0.0,
            "mean_uncertainty": float(np.mean(uncertainty[q3_mask])) if np.sum(q3_mask) > 0 else 0.0,
            "observed_prevalence": float(np.mean(y_true[q3_mask])) if np.sum(q3_mask) > 0 else 0.0,
            "description": "Model misclassified, but uncertainty correctly flagged parameter instability, routing to review.",
        },
        {
            "regime_name": "Q4: High-Confidence Silent Failure",
            "clinical_status": "Critical Failure Mode",
            "encounter_count": int(np.sum(q4_mask)),
            "cohort_share_pct": (np.sum(q4_mask) / n_total) * 100.0,
            "mean_prob": float(np.mean(y_prob[q4_mask])) if np.sum(q4_mask) > 0 else 0.0,
            "mean_uncertainty": float(np.mean(uncertainty[q4_mask])) if np.sum(q4_mask) > 0 else 0.0,
            "observed_prevalence": float(np.mean(y_true[q4_mask])) if np.sum(q4_mask) > 0 else 0.0,
            "description": "Model misclassified with falsely high confidence (low sigma_p). High-risk failure regime.",
        },
    ]

    df_regimes = pd.DataFrame(regimes)

    summary = {
        "uncertainty_threshold": u_thresh,
        "total_encounters": n_total,
        "total_errors": int(np.sum(is_error)),
        "error_rate": float(np.mean(is_error)),
        "alerted_error_capture_rate": float(np.sum(q3_mask) / np.sum(is_error)) if np.sum(is_error) > 0 else 0.0,
        "silent_failure_rate": float(np.sum(q4_mask) / np.sum(is_error)) if np.sum(is_error) > 0 else 0.0,
        "q4_high_confidence_error_count": int(np.sum(q4_mask)),
    }

    return df_regimes, summary
