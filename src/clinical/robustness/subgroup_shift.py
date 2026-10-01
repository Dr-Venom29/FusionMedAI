"""
Demographic & Intersectional Subgroup Shift Auditing (Phase C9).
Evaluates discrimination, calibration, and uncertainty across demographic strata and resampled population shifts.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from src.clinical.robustness.metrics import evaluate_full_robustness_profile


def evaluate_demographic_subgroups(
    df_test_raw: pd.DataFrame,
    y_test: np.ndarray,
    cal_prob_test: np.ndarray,
    uncertainty_test: np.ndarray,
    threshold: float = 0.20,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluate robustness and performance across individual demographic subgroups.
    """
    subgroups = {}

    # 1. Gender
    if "gender" in df_test_raw.columns:
        for g in ["Female", "Male"]:
            mask = (df_test_raw["gender"] == g).values
            if np.sum(mask) > 0:
                subgroups[f"Gender_{g}"] = ("Gender", g, mask)

    # 2. Age Groups
    if "age" in df_test_raw.columns:
        young_ages = {"[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)"}
        mid_ages = {"[50-60)", "[60-70)"}
        old_ages = {"[70-80)", "[80-90)", "[90-100)"}

        subgroups["Age_Younger_lt50"] = ("Age", "Age < 50", df_test_raw["age"].isin(young_ages).values)
        subgroups["Age_Middle_50to70"] = ("Age", "Age 50-70", df_test_raw["age"].isin(mid_ages).values)
        subgroups["Age_Older_ge70"] = ("Age", "Age >= 70", df_test_raw["age"].isin(old_ages).values)

    # 3. Race
    if "race" in df_test_raw.columns:
        for r in ["Caucasian", "AfricanAmerican", "Hispanic", "Other", "Asian"]:
            mask = (df_test_raw["race"] == r).values
            if np.sum(mask) >= 100:  # Minimum statistical sample
                subgroups[f"Race_{r}"] = ("Race", r, mask)

    rows = []
    detailed = {}

    # Nominal overall reference
    nom_metrics = evaluate_full_robustness_profile(
        y_true=y_test,
        y_prob=cal_prob_test,
        uncertainty=uncertainty_test,
        threshold=threshold,
    )

    for sg_key, (category, label, mask) in subgroups.items():
        n_sg = int(np.sum(mask))
        if n_sg < 50:
            continue

        y_sg = y_test[mask]
        p_sg = cal_prob_test[mask]
        u_sg = uncertainty_test[mask]

        metrics = evaluate_full_robustness_profile(
            y_true=y_sg,
            y_prob=p_sg,
            uncertainty=u_sg,
            threshold=threshold,
        )

        detailed[sg_key] = metrics

        rows.append({
            "subgroup_id": sg_key,
            "category": category,
            "subgroup_name": label,
            "n_samples": n_sg,
            "cohort_share_pct": (n_sg / len(y_test)) * 100.0,
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
            "sensitivity": metrics["sensitivity"],
            "specificity": metrics["specificity"],
            "ppv": metrics["ppv"],
            "npv": metrics["npv"],
            "error_detection_auroc": metrics["error_detection_auroc"],
            "aurc": metrics["aurc"],
            "error_at_80pct_coverage": metrics["error_at_80pct_coverage"],
        })

    df_subgroups = pd.DataFrame(rows)
    return df_subgroups, detailed


def evaluate_population_composition_shifts(
    df_test_raw: pd.DataFrame,
    y_test: np.ndarray,
    cal_prob_test: np.ndarray,
    uncertainty_test: np.ndarray,
    threshold: float = 0.20,
    n_resamples: int = 14913,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate macroeconomic / demographic population distribution shifts via weighted resampling.
    """
    rng = np.random.RandomState(seed)
    n_total = len(y_test)

    # Demographic indicators
    young_ages = {"[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)"}
    mid_ages = {"[50-60)", "[60-70)"}
    old_ages = {"[70-80)", "[80-90)", "[90-100)"}

    is_young = df_test_raw["age"].isin(young_ages).values
    is_mid = df_test_raw["age"].isin(mid_ages).values
    is_old = df_test_raw["age"].isin(old_ages).values

    is_female = (df_test_raw["gender"] == "Female").values
    is_male = (df_test_raw["gender"] == "Male").values

    is_cauc = (df_test_raw["race"] == "Caucasian").values
    is_aa = (df_test_raw["race"] == "AfricanAmerican").values

    # Base nominal reference metrics
    nom_metrics = evaluate_full_robustness_profile(
        y_true=y_test,
        y_prob=cal_prob_test,
        uncertainty=uncertainty_test,
        threshold=threshold,
    )

    shift_definitions = [
        ("Nominal_Reference", np.ones(n_total) / n_total, "Baseline locked test population composition"),
        (
            "Shift_Geriatric_Enriched",
            np.where(is_old, 0.70 / np.sum(is_old), np.where(is_mid, 0.20 / np.sum(is_mid), 0.10 / np.sum(is_young))),
            "Geriatric-heavy population (70% Age >= 70, 20% Age 50-70, 10% Age < 50)",
        ),
        (
            "Shift_Younger_Enriched",
            np.where(is_young, 0.50 / np.sum(is_young), np.where(is_mid, 0.35 / np.sum(is_mid), 0.15 / np.sum(is_old))),
            "Younger-skewed population (50% Age < 50, 35% Age 50-70, 15% Age >= 70)",
        ),
        (
            "Shift_AfricanAmerican_Enriched",
            np.where(is_aa, 0.50 / np.sum(is_aa), np.where(is_cauc, 0.40 / np.sum(is_cauc), 0.10 / (n_total - np.sum(is_aa) - np.sum(is_cauc)))),
            "Minority-enriched population (50% African American, 40% Caucasian, 10% Other)",
        ),
        (
            "Shift_Female_Preponderance",
            np.where(is_female, 0.75 / np.sum(is_female), 0.25 / np.sum(is_male)),
            "Female-predominant population (75% Female, 25% Male)",
        ),
    ]

    rows = []
    for name, weights, desc in shift_definitions:
        # Normalize weights
        weights = weights / np.sum(weights)
        # Resample indices
        sampled_indices = rng.choice(n_total, size=n_resamples, replace=True, p=weights)

        y_s = y_test[sampled_indices]
        p_s = cal_prob_test[sampled_indices]
        u_s = uncertainty_test[sampled_indices]

        metrics = evaluate_full_robustness_profile(
            y_true=y_s,
            y_prob=p_s,
            uncertainty=u_s,
            threshold=threshold,
        )

        rows.append({
            "shift_scenario": name,
            "description": desc,
            "n_samples": metrics["n_samples"],
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
            "error_detection_auroc": metrics["error_detection_auroc"],
            "aurc": metrics["aurc"],
            "error_at_80pct_coverage": metrics["error_at_80pct_coverage"],
        })

    return pd.DataFrame(rows)
