"""
Controlled Missingness Shift Perturbation Engines (Phase C9).
Evaluates performance and uncertainty degradation under MCAR and targeted clinical missingness.
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd
from src.clinical.modeling.preprocessing import ClinicalPreprocessor
from src.clinical.uncertainty.bootstrap_ensemble import BootstrapCatBoostEnsemble
from src.clinical.robustness.metrics import evaluate_full_robustness_profile


def apply_random_missingness(
    df: pd.DataFrame,
    fraction: float,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Apply additional random missingness (MCAR) to eligible feature columns in raw dataframe.
    """
    df_pert = df.copy()
    rng = np.random.RandomState(seed)

    # Eligible columns exclude key tracking IDs and target labels
    exclude_cols = {"encounter_id", "patient_nbr", "target_binary", "readmitted", "split", "is_cohort_eligible"}
    feature_cols = [c for c in df_pert.columns if c not in exclude_cols]

    for col in feature_cols:
        mask = rng.rand(len(df_pert)) < fraction
        if df_pert[col].dtype == object or isinstance(df_pert[col].dtype, pd.CategoricalDtype):
            df_pert.loc[mask, col] = "?"
        else:
            df_pert.loc[mask, col] = np.nan

    return df_pert


def apply_targeted_missingness(
    df: pd.DataFrame,
    target_columns: List[str],
    fraction: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Apply targeted missingness specifically to designated clinical feature columns.
    """
    df_pert = df.copy()
    rng = np.random.RandomState(seed)

    for col in target_columns:
        if col in df_pert.columns:
            if fraction >= 1.0:
                mask = np.ones(len(df_pert), dtype=bool)
            else:
                mask = rng.rand(len(df_pert)) < fraction

            if df_pert[col].dtype == object or isinstance(df_pert[col].dtype, pd.CategoricalDtype):
                df_pert.loc[mask, col] = "?"
            else:
                df_pert.loc[mask, col] = np.nan

    return df_pert


def evaluate_missingness_scenarios(
    df_test_raw: pd.DataFrame,
    preprocessor: ClinicalPreprocessor,
    ensemble: BootstrapCatBoostEnsemble,
    calibrator: Any,
    threshold: float = 0.20,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run the suite of missingness shift experiments:
    M0: Nominal (0% added missingness)
    M1: +10% Random MCAR
    M2: +25% Random MCAR
    M3: +50% Random MCAR
    M4a: Targeted Glycemic/Lab Missingness (A1Cresult, max_glu_serum, num_lab_procedures)
    M4b: Targeted Medication Missingness (all diabetic medications & change)
    M4c: Targeted Utilization/Complexity Missingness (inpatient, emergency, outpatient, hospital time)
    """
    scenarios = [
        ("M0_Nominal_Reference", lambda df: df.copy(), "Nominal test distribution with pre-existing missingness"),
        ("M1_Random_MCAR_10pct", lambda df: apply_random_missingness(df, 0.10, seed=seed), "Random 10% MCAR masking across all features"),
        ("M2_Random_MCAR_25pct", lambda df: apply_random_missingness(df, 0.25, seed=seed), "Random 25% MCAR masking across all features"),
        ("M3_Random_MCAR_50pct", lambda df: apply_random_missingness(df, 0.50, seed=seed), "Random 50% MCAR masking across all features"),
        (
            "M4a_Targeted_Glycemic_Lab",
            lambda df: apply_targeted_missingness(df, ["A1Cresult", "max_glu_serum", "num_lab_procedures"], fraction=1.0, seed=seed),
            "Complete masking of A1C, serum glucose, and lab procedure volume",
        ),
        (
            "M4b_Targeted_Medications",
            lambda df: apply_targeted_missingness(
                df,
                ["insulin", "metformin", "glipizide", "glyburide", "pioglitazone", "rosiglitazone", "glimepiride", "change", "diabetesMed"],
                fraction=1.0,
                seed=seed,
            ),
            "Complete masking of all diabetic medication therapy features",
        ),
        (
            "M4c_Targeted_Utilization",
            lambda df: apply_targeted_missingness(
                df,
                ["number_inpatient", "number_emergency", "number_outpatient", "time_in_hospital", "num_procedures"],
                fraction=1.0,
                seed=seed,
            ),
            "Complete masking of prior utilization and stay duration features",
        ),
    ]

    rows = []
    detailed_results = {}
    nominal_metrics = None

    for name, transform_fn, desc in scenarios:
        df_pert = transform_fn(df_test_raw)
        X_pert, y_pert, _ = preprocessor.transform(df_pert)

        dist = ensemble.predict_distribution(X_pert)
        raw_prob = dist["mean_prob"]
        uncertainty = dist["std_prob"]

        if calibrator is not None:
            cal_prob = calibrator.predict_proba(raw_prob)
        else:
            cal_prob = raw_prob

        metrics = evaluate_full_robustness_profile(
            y_true=y_pert,
            y_prob=cal_prob,
            uncertainty=uncertainty,
            threshold=threshold,
        )

        detailed_results[name] = {
            "metrics": metrics,
            "description": desc,
            "probabilities": cal_prob,
            "uncertainty": uncertainty,
            "y_true": y_pert,
        }

        if name == "M0_Nominal_Reference":
            nominal_metrics = metrics

        delta_roc = metrics["roc_auc"] - nominal_metrics["roc_auc"]
        delta_pr = metrics["pr_auc"] - nominal_metrics["pr_auc"]
        delta_brier = metrics["brier_score"] - nominal_metrics["brier_score"]
        delta_ece = metrics["ece"] - nominal_metrics["ece"]
        delta_slope = metrics["calibration_slope"] - nominal_metrics["calibration_slope"]
        delta_unc = metrics["mean_uncertainty"] - nominal_metrics["mean_uncertainty"]
        delta_err = metrics["error_rate"] - nominal_metrics["error_rate"]

        rows.append({
            "scenario": name,
            "description": desc,
            "n_samples": metrics["n_samples"],
            "roc_auc": metrics["roc_auc"],
            "delta_roc_auc": delta_roc,
            "pr_auc": metrics["pr_auc"],
            "delta_pr_auc": delta_pr,
            "brier_score": metrics["brier_score"],
            "delta_brier": delta_brier,
            "ece": metrics["ece"],
            "delta_ece": delta_ece,
            "calibration_slope": metrics["calibration_slope"],
            "delta_slope": delta_slope,
            "mean_uncertainty": metrics["mean_uncertainty"],
            "delta_mean_uncertainty": delta_unc,
            "p95_uncertainty": metrics["p95_uncertainty"],
            "error_rate": metrics["error_rate"],
            "delta_error_rate": delta_err,
            "error_detection_auroc": metrics["error_detection_auroc"],
            "aurc": metrics["aurc"],
            "error_at_80pct_coverage": metrics["error_at_80pct_coverage"],
        })

    df_results = pd.DataFrame(rows)
    return df_results, detailed_results
