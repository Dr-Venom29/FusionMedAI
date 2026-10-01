"""
Subgroup calibration auditing for clinical readmission models.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from src.clinical.calibration.metrics import evaluate_calibration_metrics


def evaluate_subgroup_calibration(
    df_features: pd.DataFrame,
    y_true: np.ndarray,
    y_prob_raw: np.ndarray,
    y_prob_cal: np.ndarray,
) -> pd.DataFrame:
    """
    Evaluate calibration metrics stratified across clinically relevant subgroups:
    1. Prior Inpatient Utilization (0 vs >= 1)
    2. Gender (Female vs Male)
    3. Age Cohorts (<50, 50-70, >=70)
    """
    y_true = np.asarray(y_true).ravel()
    y_prob_raw = np.asarray(y_prob_raw).ravel()
    y_prob_cal = np.asarray(y_prob_cal).ravel()

    subgroups = {}

    # 1. Prior Inpatient Utilization
    if "number_inpatient" in df_features.columns:
        inpatient = df_features["number_inpatient"].values
        subgroups["Inpatient = 0"] = inpatient == 0
        subgroups["Inpatient >= 1"] = inpatient >= 1

    # 2. Gender
    if "gender_Male" in df_features.columns:
        is_male = df_features["gender_Male"].values == 1
        subgroups["Gender: Male"] = is_male
        subgroups["Gender: Female"] = ~is_male
    elif "gender" in df_features.columns:
        subgroups["Gender: Male"] = df_features["gender"].values == "Male"
        subgroups["Gender: Female"] = df_features["gender"].values == "Female"

    # 3. Age cohorts
    if "age_ordinal" in df_features.columns:
        age = df_features["age_ordinal"].values
        # Ordinal age: [0-10)=0, [10-20)=1, ..., [50-60)=5, [60-70)=6, [70-80)=7, etc.
        subgroups["Age < 50"] = age < 5
        subgroups["Age 50-70"] = (age >= 5) & (age < 7)
        subgroups["Age >= 70"] = age >= 7

    results = []

    for name, mask in subgroups.items():
        if np.sum(mask) == 0:
            continue
        n_sub = int(np.sum(mask))
        y_sub_true = y_true[mask]
        y_sub_raw = y_prob_raw[mask]
        y_sub_cal = y_prob_cal[mask]

        prev = float(np.mean(y_sub_true))

        # Evaluate Raw
        raw_m = evaluate_calibration_metrics(y_sub_true, y_sub_raw)
        # Evaluate Calibrated
        cal_m = evaluate_calibration_metrics(y_sub_true, y_sub_cal)

        results.append({
            "subgroup": name,
            "n_samples": n_sub,
            "prevalence": prev,
            "raw_brier": raw_m["brier_score"],
            "cal_brier": cal_m["brier_score"],
            "raw_log_loss": raw_m["log_loss"],
            "cal_log_loss": cal_m["log_loss"],
            "raw_ece": raw_m["ece"],
            "cal_ece": cal_m["ece"],
            "raw_intercept": raw_m["calibration_intercept"],
            "cal_intercept": cal_m["calibration_intercept"],
            "raw_slope": raw_m["calibration_slope"],
            "cal_slope": cal_m["calibration_slope"],
            "roc_auc": cal_m["roc_auc"],
            "pr_auc": cal_m["pr_auc"],
        })

    return pd.DataFrame(results)
