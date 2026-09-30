"""Evaluation and Error Analysis Module for Clinical Baseline Models.

Conducts:
- Multi-model comparison across validation and test sets
- Comprehensive threshold performance evaluation
- Subgroup-stratified error analysis (False Positives, False Negatives, Subgroup Sensitivity/Specificity)
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from src.clinical.modeling.metrics import (
    compute_classification_metrics,
    compute_threshold_metrics,
    compute_calibration_metrics,
)
from src.clinical.modeling.schema import map_icd9_to_chapter


def evaluate_model_on_split(
    model,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    thresholds: List[float] = [0.10, 0.20, 0.30, 0.40, 0.50],
) -> Dict[str, Any]:
    """Compute complete evaluation payload for a single model on a dataset partition."""
    y_prob = model.predict_proba(X)[:, 1]
    
    # Base summary at threshold=0.5 and clinical threshold=0.20
    summary_50 = compute_classification_metrics(y, y_prob, threshold=0.50, model_name=model_name)
    summary_20 = compute_classification_metrics(y, y_prob, threshold=0.20, model_name=model_name)
    
    # Threshold sweep
    threshold_results = compute_threshold_metrics(y, y_prob, thresholds=thresholds, model_name=model_name)
    
    # Calibration metrics
    calibration_results = compute_calibration_metrics(y, y_prob, n_bins=10, model_name=model_name)
    
    return {
        "model": model_name,
        "summary_default_th50": summary_50,
        "summary_clinical_th20": summary_20,
        "threshold_evaluations": threshold_results,
        "calibration": calibration_results,
    }


def compute_stratified_error_analysis(
    df_raw: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.20,
    model_name: str = "model",
) -> Dict[str, Any]:
    """Perform subgroup-stratified error analysis on false positives and false negatives."""
    y_pred = (y_prob >= threshold).astype(int)
    df_eval = df_raw.copy()
    df_eval["y_true"] = y_true
    df_eval["y_prob"] = y_prob
    df_eval["y_pred"] = y_pred
    df_eval["is_tp"] = (df_eval["y_true"] == 1) & (df_eval["y_pred"] == 1)
    df_eval["is_fp"] = (df_eval["y_true"] == 0) & (df_eval["y_pred"] == 1)
    df_eval["is_tn"] = (df_eval["y_true"] == 0) & (df_eval["y_pred"] == 0)
    df_eval["is_fn"] = (df_eval["y_true"] == 1) & (df_eval["y_pred"] == 0)

    # Derived subgroup columns for stratification
    df_eval["primary_diag_chapter"] = df_eval["diag_1"].apply(map_icd9_to_chapter)
    df_eval["inpatient_prior_group"] = df_eval["number_inpatient"].apply(
        lambda x: "0" if x == 0 else ("1" if x == 1 else ("2" if x == 2 else ">=3"))
    )

    dimensions = {
        "age": "age",
        "race": "race",
        "gender": "gender",
        "primary_diag_chapter": "primary_diag_chapter",
        "inpatient_prior_group": "inpatient_prior_group",
        "insulin": "insulin",
        "change": "change",
    }

    subgroup_reports = {}
    for dim_name, col in dimensions.items():
        if col not in df_eval.columns:
            continue
        strata = []
        for group_val, grp in df_eval.groupby(col, observed=True):
            n_grp = len(grp)
            n_pos = int(grp["y_true"].sum())
            n_tp = int(grp["is_tp"].sum())
            n_fp = int(grp["is_fp"].sum())
            n_tn = int(grp["is_tn"].sum())
            n_fn = int(grp["is_fn"].sum())
            
            sens = round(n_tp / (n_tp + n_fn), 4) if (n_tp + n_fn) > 0 else 0.0
            spec = round(n_tn / (n_tn + n_fp), 4) if (n_tn + n_fp) > 0 else 0.0
            ppv = round(n_tp / (n_tp + n_fp), 4) if (n_tp + n_fp) > 0 else 0.0
            
            strata.append({
                "subgroup": str(group_val),
                "total_encounters": n_grp,
                "observed_prevalence": round(n_pos / n_grp, 4) if n_grp > 0 else 0.0,
                "tp": n_tp,
                "fp": n_fp,
                "tn": n_tn,
                "fn": n_fn,
                "sensitivity": sens,
                "specificity": spec,
                "ppv": ppv,
                "fp_rate_within_negatives": round(n_fp / (n_tn + n_fp), 4) if (n_tn + n_fp) > 0 else 0.0,
                "fn_rate_within_positives": round(n_fn / (n_tp + n_fn), 4) if (n_tp + n_fn) > 0 else 0.0,
            })
        subgroup_reports[dim_name] = strata

    return {
        "model": model_name,
        "operating_threshold": threshold,
        "total_evaluated": len(df_eval),
        "overall_prevalence": round(float(np.mean(y_true)), 4),
        "subgroups": subgroup_reports,
    }
