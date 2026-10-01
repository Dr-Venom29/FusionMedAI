"""
Encounter Complexity & Utilization Phenotype Shift Auditing (Phase C9).
Evaluates model stability and uncertainty when encounter mixtures diverge from the nominal training distribution.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from src.clinical.robustness.metrics import evaluate_full_robustness_profile


def evaluate_utilization_phenotype_strata(
    df_test_raw: pd.DataFrame,
    y_test: np.ndarray,
    cal_prob_test: np.ndarray,
    uncertainty_test: np.ndarray,
    threshold: float = 0.20,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluate robustness across natural clinical utilization and complexity strata.
    """
    # Nominal reference
    nom_metrics = evaluate_full_robustness_profile(
        y_true=y_test,
        y_prob=cal_prob_test,
        uncertainty=uncertainty_test,
        threshold=threshold,
    )

    strata_masks = {
        "Util_Zero_Inpatient": (
            "Prior Utilization",
            "Zero Inpatient Visits (0)",
            (df_test_raw["number_inpatient"] == 0).values,
        ),
        "Util_Moderate_Inpatient": (
            "Prior Utilization",
            "Moderate Inpatient (1-2 visits)",
            df_test_raw["number_inpatient"].isin([1, 2]).values,
        ),
        "Util_Frequent_Inpatient": (
            "Prior Utilization",
            "High Inpatient (>= 3 visits)",
            (df_test_raw["number_inpatient"] >= 3).values,
        ),
        "Complex_High_Diagnoses": (
            "Multimorbidity",
            "High Diagnoses (>= 9 codes)",
            (df_test_raw["number_diagnoses"] >= 9).values,
        ),
        "Complex_Low_Diagnoses": (
            "Multimorbidity",
            "Low Diagnoses (<= 5 codes)",
            (df_test_raw["number_diagnoses"] <= 5).values,
        ),
        "Complex_Polypharmacy": (
            "Pharmacotherapy",
            "High Polypharmacy (>= 20 meds)",
            (df_test_raw["num_medications"] >= 20).values,
        ),
        "Complex_Extended_Stay": (
            "Acuity / Stay",
            "Extended Length of Stay (>= 7 days)",
            (df_test_raw["time_in_hospital"] >= 7).values,
        ),
    }

    rows = []
    detailed = {}

    for k, (cat, label, mask) in strata_masks.items():
        n_s = int(np.sum(mask))
        if n_s < 50:
            continue

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
            "stratum_id": k,
            "category": cat,
            "stratum_name": label,
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


def evaluate_encounter_composition_shifts(
    df_test_raw: pd.DataFrame,
    y_test: np.ndarray,
    cal_prob_test: np.ndarray,
    uncertainty_test: np.ndarray,
    threshold: float = 0.20,
    n_resamples: int = 14913,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate systemic encounter mixture shifts via weighted resampling.
    """
    rng = np.random.RandomState(seed)
    n_total = len(y_test)

    inp_0 = (df_test_raw["number_inpatient"] == 0).values
    inp_ge1 = (df_test_raw["number_inpatient"] >= 1).values
    high_diag = (df_test_raw["number_diagnoses"] >= 9).values
    low_diag = (df_test_raw["number_diagnoses"] <= 5).values
    short_stay = (df_test_raw["time_in_hospital"] <= 3).values

    nom_metrics = evaluate_full_robustness_profile(
        y_true=y_test,
        y_prob=cal_prob_test,
        uncertainty=uncertainty_test,
        threshold=threshold,
    )

    shift_definitions = [
        ("Nominal_Reference", np.ones(n_total) / n_total, "Baseline locked test encounter composition"),
        (
            "Shift_High_Utilization_Heavy",
            np.where(inp_ge1, 0.60 / np.sum(inp_ge1), 0.40 / np.sum(inp_0)),
            "High-utilization heavy (60% prior inpatient >= 1, 40% prior inpatient = 0)",
        ),
        (
            "Shift_First_Time_Enriched",
            np.where(inp_0, 0.90 / np.sum(inp_0), 0.10 / np.sum(inp_ge1)),
            "First-time admission enriched (90% prior inpatient = 0, 10% prior inpatient >= 1)",
        ),
        (
            "Shift_Multimorbidity_Heavy",
            np.where(high_diag, 0.70 / np.sum(high_diag), 0.30 / (n_total - np.sum(high_diag))),
            "High multimorbidity heavy (70% Diagnoses >= 9, 30% Diagnoses < 9)",
        ),
        (
            "Shift_Low_Complexity_Heavy",
            np.where(short_stay & low_diag, 0.60 / np.sum(short_stay & low_diag), 0.40 / (n_total - np.sum(short_stay & low_diag))),
            "Low acuity / short stay heavy (60% stay <= 3 days & diagnoses <= 5)",
        ),
    ]

    rows = []
    for name, weights, desc in shift_definitions:
        weights = weights / np.sum(weights)
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
