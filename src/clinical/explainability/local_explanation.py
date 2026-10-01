"""Local case explanations and error-focused SHAP analysis."""

from typing import List, Dict, Any, Tuple
from pathlib import Path
import numpy as np
import pandas as pd


def generate_representative_local_cases(
    shap_values: np.ndarray,
    probs: np.ndarray,
    y_true: np.ndarray,
    feature_names: List[str],
    base_value: float,
    threshold: float = 0.20,
) -> pd.DataFrame:
    """Generate local SHAP explanations for representative cohorts:

    - True Positive (Correct high-risk alert above threshold)
    - True Negative (Correct low-risk baseline below threshold)
    - False Positive (High-complexity patient who avoided readmission)
    - False Negative (Subtle missed readmission below threshold)
    - Critical Risk (Top percentile highest risk prediction)
    - Lowest Risk (Bottom percentile lowest risk prediction)
    """
    eval_df = pd.DataFrame({
        "encounter_idx": np.arange(len(y_true)),
        "actual_readmitted": y_true,
        "predicted_prob": probs,
    })

    cases = []

    # 1. Critical Risk (Highest probability)
    top_risk_idx = eval_df.nlargest(3, "predicted_prob")["encounter_idx"].tolist()
    for idx in top_risk_idx:
        cases.append(("Critical Risk (p >= 0.35)", idx))

    # 2. True Positives near operating threshold (p in [0.20, 0.30], actual=1)
    tp_idx = eval_df[(eval_df["predicted_prob"] >= threshold) & (eval_df["actual_readmitted"] == 1)].head(3)["encounter_idx"].tolist()
    for idx in tp_idx:
        cases.append(("True Positive (Alert Threshold)", idx))

    # 3. False Positives (p in [0.20, 0.30], actual=0)
    fp_idx = eval_df[(eval_df["predicted_prob"] >= threshold) & (eval_df["actual_readmitted"] == 0)].head(3)["encounter_idx"].tolist()
    for idx in fp_idx:
        cases.append(("False Positive (High Complexity / Averted)", idx))

    # 4. False Negatives (p in [0.10, 0.15], actual=1)
    fn_idx = eval_df[(eval_df["predicted_prob"] < 0.15) & (eval_df["actual_readmitted"] == 1)].head(3)["encounter_idx"].tolist()
    for idx in fn_idx:
        cases.append(("False Negative (Missed Relapse)", idx))

    # 5. Lowest Risk True Negatives
    low_idx = eval_df[eval_df["actual_readmitted"] == 0].nsmallest(3, "predicted_prob")["encounter_idx"].tolist()
    for idx in low_idx:
        cases.append(("Lowest Risk Baseline", idx))

    rows = []
    for category, idx in cases:
        p_val = float(probs[idx])
        y_act = int(y_true[idx])
        s_row = shap_values[idx]

        top_pos_idx = np.argsort(s_row)[-4:][::-1]
        top_neg_idx = np.argsort(s_row)[:4]

        pos_strs = [f"{feature_names[i]} (+{s_row[i]:.4f})" for i in top_pos_idx if s_row[i] > 0]
        neg_strs = [f"{feature_names[i]} ({s_row[i]:.4f})" for i in top_neg_idx if s_row[i] < 0]

        rows.append({
            "cohort_category": category,
            "encounter_idx": idx,
            "actual_readmitted": y_act,
            "predicted_probability": round(p_val, 4),
            "base_value_log_odds": round(base_value, 4),
            "top_risk_increasing_features": "; ".join(pos_strs),
            "top_risk_decreasing_features": "; ".join(neg_strs),
        })

    return pd.DataFrame(rows)


def compute_error_case_attribution(
    shap_values: np.ndarray,
    probs: np.ndarray,
    y_true: np.ndarray,
    feature_names: List[str],
    threshold: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute average SHAP contributions for False Positives vs False Negatives."""
    fp_mask = (probs >= threshold) & (y_true == 0)
    fn_mask = (probs < threshold) & (y_true == 1)

    fp_shap_mean = np.mean(shap_values[fp_mask], axis=0) if np.sum(fp_mask) > 0 else np.zeros(len(feature_names))
    fn_shap_mean = np.mean(shap_values[fn_mask], axis=0) if np.sum(fn_mask) > 0 else np.zeros(len(feature_names))

    fp_df = pd.DataFrame({
        "feature": feature_names,
        "mean_fp_shap": fp_shap_mean,
        "mean_abs_fp_shap": np.abs(fp_shap_mean),
    }).sort_values("mean_abs_fp_shap", ascending=False).reset_index(drop=True)
    fp_df["rank_fp"] = np.arange(1, len(feature_names) + 1)

    fn_df = pd.DataFrame({
        "feature": feature_names,
        "mean_fn_shap": fn_shap_mean,
        "mean_abs_fn_shap": np.abs(fn_shap_mean),
    }).sort_values("mean_abs_fn_shap", ascending=False).reset_index(drop=True)
    fn_df["rank_fn"] = np.arange(1, len(feature_names) + 1)

    return fp_df, fn_df
