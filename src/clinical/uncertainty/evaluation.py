"""
Evaluation routines for Clinical Prediction Uncertainty (Phase C8).
Includes Error Detection, Risk-Coverage (AURC / E-AURC), Threshold Tiers, Subgroup Audits, and Convergence Analysis.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score


def evaluate_error_detection(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    uncertainty: np.ndarray,
    threshold: float = 0.20,
) -> Dict[str, Any]:
    """
    Evaluate whether predictive uncertainty discriminates between correct and incorrect predictions.
    
    Binary error definition: e_i = 1 if (y_prob >= threshold) != y_true else 0.
    """
    y_pred = (y_prob >= threshold).astype(int)
    errors = (y_pred != y_true).astype(int)
    error_rate = float(np.mean(errors))

    # Error detection AUROC and AUPRC
    if np.sum(errors) > 0 and np.sum(errors) < len(errors):
        error_auroc = float(roc_auc_score(errors, uncertainty))
        error_auprc = float(average_precision_score(errors, uncertainty))
    else:
        error_auroc = 0.5
        error_auprc = float(np.mean(errors))

    unc_correct = uncertainty[errors == 0]
    unc_incorrect = uncertainty[errors == 1]

    return {
        "threshold": threshold,
        "total_encounters": len(y_true),
        "error_count": int(np.sum(errors)),
        "error_rate": error_rate,
        "error_auroc": error_auroc,
        "error_auprc": error_auprc,
        "mean_unc_correct": float(np.mean(unc_correct)) if len(unc_correct) > 0 else 0.0,
        "median_unc_correct": float(np.median(unc_correct)) if len(unc_correct) > 0 else 0.0,
        "std_unc_correct": float(np.std(unc_correct)) if len(unc_correct) > 0 else 0.0,
        "mean_unc_incorrect": float(np.mean(unc_incorrect)) if len(unc_incorrect) > 0 else 0.0,
        "median_unc_incorrect": float(np.median(unc_incorrect)) if len(unc_incorrect) > 0 else 0.0,
        "std_unc_incorrect": float(np.std(unc_incorrect)) if len(unc_incorrect) > 0 else 0.0,
    }


def evaluate_risk_coverage(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    uncertainty: np.ndarray,
    threshold: float = 0.20,
    coverages: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Perform Risk-Coverage analysis by progressively rejecting the most uncertain predictions.
    
    Computes AURC (Area Under Risk-Coverage Curve), Optimal AURC, and Excess AURC (E-AURC).
    """
    if coverages is None:
        coverages = np.linspace(1.0, 0.10, 19)

    n_samples = len(y_true)
    y_pred = (y_prob >= threshold).astype(int)
    errors = (y_pred != y_true).astype(int)
    overall_error_rate = float(np.mean(errors))

    # Sort indices by uncertainty ascending (lowest uncertainty / highest confidence first)
    sorted_indices = np.argsort(uncertainty)

    rows = []
    for cov in coverages:
        k = max(1, int(np.round(cov * n_samples)))
        retained_indices = sorted_indices[:k]
        cov_error_rate = float(np.mean(errors[retained_indices]))
        retained_y_true = y_true[retained_indices]
        retained_y_pred = y_pred[retained_indices]
        tp = int(np.sum((retained_y_true == 1) & (retained_y_pred == 1)))
        fp = int(np.sum((retained_y_true == 0) & (retained_y_pred == 1)))
        tn = int(np.sum((retained_y_true == 0) & (retained_y_pred == 0)))
        fn = int(np.sum((retained_y_true == 1) & (retained_y_pred == 0)))

        rows.append({
            "coverage": float(cov),
            "retained_count": int(k),
            "rejected_count": int(n_samples - k),
            "risk_error_rate": cov_error_rate,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "mean_uncertainty_retained": float(np.mean(uncertainty[retained_indices])),
        })

    df_rc = pd.DataFrame(rows)

    # Compute empirical AURC via trapezoidal integration over coverage [0.10, 1.0] normalized
    # Standard AURC is integral of risk(c) dc from c=0 to 1
    # Here we integrate over evaluated coverages:
    cov_arr = df_rc["coverage"].values
    risk_arr = df_rc["risk_error_rate"].values
    
    # Sort coverage ascending for integration
    sort_idx = np.argsort(cov_arr)
    c_sorted = cov_arr[sort_idx]
    r_sorted = risk_arr[sort_idx]
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    aurc = float(trapz_fn(r_sorted, c_sorted))

    # Optimal Oracle AURC: oracle rejects true errors first
    # For a perfect ranker, error rate is 0 until coverage exceeds (1 - error_rate), then rises to overall_error_rate
    # Oracle risk at coverage c <= (1 - E): risk = 0
    # Oracle risk at coverage c > (1 - E): risk = (c - (1 - E)) / c
    c_fine = np.linspace(0.01, 1.0, 1000)
    oracle_risk = np.where(c_fine <= (1.0 - overall_error_rate), 0.0, (c_fine - (1.0 - overall_error_rate)) / c_fine)
    optimal_aurc = float(trapz_fn(oracle_risk, c_fine))
    e_aurc = float(aurc - optimal_aurc)

    summary = {
        "baseline_error_rate": overall_error_rate,
        "aurc": aurc,
        "optimal_aurc": optimal_aurc,
        "e_aurc": e_aurc,
        "error_at_80pct_coverage": float(df_rc.loc[np.isclose(df_rc["coverage"], 0.80, atol=0.03), "risk_error_rate"].values[0]) if any(np.isclose(df_rc["coverage"], 0.80, atol=0.03)) else overall_error_rate,
        "error_at_50pct_coverage": float(df_rc.loc[np.isclose(df_rc["coverage"], 0.50, atol=0.03), "risk_error_rate"].values[0]) if any(np.isclose(df_rc["coverage"], 0.50, atol=0.03)) else overall_error_rate,
    }

    return df_rc, summary


def evaluate_threshold_uncertainty_tiers(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    uncertainty: np.ndarray,
    threshold: float = 0.20,
    uncertainty_pct_threshold: float = 75.0,
) -> pd.DataFrame:
    """
    Categorize encounters into clinical decision-support tiers based on risk probability and uncertainty.
    """
    unc_cutoff = float(np.percentile(uncertainty, uncertainty_pct_threshold))
    near_threshold_margin = 0.03

    y_pred = (y_prob >= threshold).astype(int)
    errors = (y_pred != y_true).astype(int)

    is_high_risk = (y_prob >= threshold)
    is_low_risk = (y_prob < threshold)
    is_high_unc = (uncertainty >= unc_cutoff)
    is_low_unc = (uncertainty < unc_cutoff)
    is_near_thresh = (np.abs(y_prob - threshold) <= near_threshold_margin)

    tiers = {
        "Low Risk / Low Uncertainty": is_low_risk & is_low_unc & ~is_near_thresh,
        "Low Risk / High Uncertainty": is_low_risk & is_high_unc & ~is_near_thresh,
        "High Risk / Low Uncertainty": is_high_risk & is_low_unc & ~is_near_thresh,
        "High Risk / High Uncertainty": is_high_risk & is_high_unc & ~is_near_thresh,
        "Near Threshold / High Uncertainty (Ambiguity)": is_near_thresh & is_high_unc,
        "Near Threshold / Low Uncertainty": is_near_thresh & is_low_unc,
    }

    rows = []
    total_n = len(y_true)
    for tier_name, mask in tiers.items():
        count = int(np.sum(mask))
        pct = (count / total_n) * 100.0 if total_n > 0 else 0.0
        tier_errors = errors[mask]
        tier_y = y_true[mask]
        tier_p = y_prob[mask]
        tier_u = uncertainty[mask]

        rows.append({
            "tier": tier_name,
            "encounter_count": count,
            "percentage": pct,
            "observed_prevalence": float(np.mean(tier_y)) if count > 0 else 0.0,
            "mean_predicted_prob": float(np.mean(tier_p)) if count > 0 else 0.0,
            "mean_uncertainty": float(np.mean(tier_u)) if count > 0 else 0.0,
            "error_rate": float(np.mean(tier_errors)) if count > 0 else 0.0,
        })

    return pd.DataFrame(rows)


def evaluate_subgroup_uncertainty(
    df_raw: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    uncertainty: np.ndarray,
    threshold: float = 0.20,
) -> pd.DataFrame:
    """
    Audit uncertainty distributions and error-detection performance across clinical and demographic subgroups.
    """
    subgroups = {}

    # Prior Inpatient
    if "number_inpatient" in df_raw.columns:
        subgroups["Prior Inpatient = 0"] = (df_raw["number_inpatient"].values == 0)
        subgroups["Prior Inpatient >= 1"] = (df_raw["number_inpatient"].values >= 1)

    # Gender
    if "gender" in df_raw.columns:
        subgroups["Gender: Male"] = (df_raw["gender"].values == "Male")
        subgroups["Gender: Female"] = (df_raw["gender"].values == "Female")

    # Age brackets
    if "age" in df_raw.columns:
        subgroups["Age < 50"] = df_raw["age"].isin(["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)"]).values
        subgroups["Age 50-70"] = df_raw["age"].isin(["[50-60)", "[60-70)"]).values
        subgroups["Age >= 70"] = df_raw["age"].isin(["[70-80)", "[80-90)", "[90-100)"]).values

    rows = []
    for sub_name, mask in subgroups.items():
        sub_n = int(np.sum(mask))
        if sub_n == 0:
            continue

        sub_y = y_true[mask]
        sub_p = y_prob[mask]
        sub_u = uncertainty[mask]
        sub_pred = (sub_p >= threshold).astype(int)
        sub_errors = (sub_pred != sub_y).astype(int)

        if np.sum(sub_errors) > 0 and np.sum(sub_errors) < sub_n:
            sub_err_auroc = float(roc_auc_score(sub_errors, sub_u))
        else:
            sub_err_auroc = 0.5

        rows.append({
            "subgroup": sub_name,
            "n_samples": sub_n,
            "prevalence": float(np.mean(sub_y)),
            "mean_uncertainty": float(np.mean(sub_u)),
            "median_uncertainty": float(np.median(sub_u)),
            "std_uncertainty": float(np.std(sub_u)),
            "q25_uncertainty": float(np.percentile(sub_u, 25)),
            "q75_uncertainty": float(np.percentile(sub_u, 75)),
            "error_rate": float(np.mean(sub_errors)),
            "error_detection_auroc": sub_err_auroc,
        })

    return pd.DataFrame(rows)


def evaluate_convergence(
    all_probs: np.ndarray,
    y_true: np.ndarray,
    subset_sizes: Optional[List[int]] = None,
    threshold: float = 0.20,
) -> pd.DataFrame:
    """
    Evaluate ensemble stability, correlation, and error-detection AUROC as ensemble size M varies.
    """
    total_members = all_probs.shape[1]
    if subset_sizes is None:
        subset_sizes = [m for m in [5, 10, 20, 30, 40, 50, total_members] if m <= total_members]
        subset_sizes = sorted(list(set(subset_sizes)))

    full_mean = np.mean(all_probs, axis=1)
    full_var = np.var(all_probs, axis=1, ddof=1)
    full_std = np.sqrt(full_var)

    rows = []
    for m in subset_sizes:
        sub_probs = all_probs[:, :m]
        sub_mean = np.mean(sub_probs, axis=1)
        sub_var = np.var(sub_probs, axis=1, ddof=1 if m > 1 else 0)
        sub_std = np.sqrt(sub_var)

        # Ranking correlation with full ensemble
        corr_mean, _ = spearmanr(sub_mean, full_mean)
        corr_std, _ = spearmanr(sub_std, full_std) if m > 1 else (0.0, 0.0)

        # Mean Absolute Deviation from full ensemble
        mad_mean = float(np.mean(np.abs(sub_mean - full_mean)))
        mad_std = float(np.mean(np.abs(sub_std - full_std))) if m > 1 else float(np.mean(full_std))

        # Error detection AUROC for subset
        sub_errors = ((sub_mean >= threshold).astype(int) != y_true).astype(int)
        if m > 1 and np.sum(sub_errors) > 0 and np.sum(sub_errors) < len(y_true):
            sub_err_auroc = float(roc_auc_score(sub_errors, sub_std))
        else:
            sub_err_auroc = 0.5

        rows.append({
            "ensemble_size_M": int(m),
            "spearman_corr_mean": float(corr_mean),
            "spearman_corr_std": float(corr_std),
            "mad_mean_vs_full": mad_mean,
            "mad_std_vs_full": mad_std,
            "error_detection_auroc": sub_err_auroc,
            "mean_uncertainty": float(np.mean(sub_std)),
        })

    return pd.DataFrame(rows)
