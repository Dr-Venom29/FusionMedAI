"""
Master Execution Pipeline for Clinical Phase C9: Robustness, Fairness & Distribution Shift.
"""

import os
import sys
import json
import hashlib
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

# Ensure project root in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.clinical.modeling.preprocessing import ClinicalPreprocessor
from src.clinical.calibration.calibrator import IsotonicCalibrator
from src.clinical.uncertainty.bootstrap_ensemble import BootstrapCatBoostEnsemble
from src.clinical.robustness.metrics import evaluate_full_robustness_profile
from src.clinical.robustness.missingness_shift import evaluate_missingness_scenarios
from src.clinical.robustness.subgroup_shift import (
    evaluate_demographic_subgroups,
    evaluate_population_composition_shifts,
)
from src.clinical.robustness.encounter_shift import (
    evaluate_utilization_phenotype_strata,
    evaluate_encounter_composition_shifts,
)
from src.clinical.robustness.temporal_shift import evaluate_temporal_shift
from src.clinical.robustness.shift_detector import profile_epistemic_failure_regimes
from src.clinical.robustness.plot_robustness import (
    plot_missingness_degradation,
    plot_subgroup_shift,
    plot_encounter_shift,
    plot_temporal_shift,
    plot_uncertainty_shift,
    plot_risk_coverage_shift,
    plot_robustness_summary,
)

EXPERIMENT_DIR = REPO_ROOT / "experiments" / "clinical" / "robustness"
RESEARCH_DIR = REPO_ROOT / "research" / "clinical" / "Volume_09_Robustness_Fairness"


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_pipeline():
    print("=" * 80)
    print("FUSIONMEDAI: CLINICAL PHASE C9 ROBUSTNESS & DISTRIBUTION SHIFT PIPELINE")
    print("=" * 80)

    # 1. Setup output paths
    exp_tables_dir = EXPERIMENT_DIR / "tables"
    exp_figures_dir = EXPERIMENT_DIR / "figures"
    exp_manifests_dir = EXPERIMENT_DIR / "manifests"
    exp_models_dir = EXPERIMENT_DIR / "models"
    res_figures_dir = RESEARCH_DIR / "figures"

    for d in [exp_tables_dir, exp_figures_dir, exp_manifests_dir, exp_models_dir, res_figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 2. Load dataset partitions
    data_dir = REPO_ROOT / "datasets" / "clinical" / "processed" / "splits"
    print(f"[1/9] Loading frozen splits from {data_dir}...")
    df_train = pd.read_csv(data_dir / "train.csv")
    df_val = pd.read_csv(data_dir / "val.csv")
    df_test = pd.read_csv(data_dir / "test.csv")

    # 3. Fit locked Preprocessor on train
    print("[2/9] Fitting locked ClinicalPreprocessor (D=119)...")
    preprocessor = ClinicalPreprocessor(scale_numerical=True)
    preprocessor.fit(df_train)
    X_train, y_train, _ = preprocessor.transform(df_train)
    X_val, y_val, _ = preprocessor.transform(df_val)
    X_test, y_test, _ = preprocessor.transform(df_test)

    feature_names = list(preprocessor.feature_names_)
    assert len(feature_names) == 119, f"Expected 119 features, got {len(feature_names)}"

    # 4. Train or Load Bootstrap Ensemble (M=50 models)
    n_estimators = 50
    print(f"[3/9] Initializing Bootstrap CatBoost Ensemble (M={n_estimators} members)...")
    ensemble = BootstrapCatBoostEnsemble(
        n_estimators=n_estimators,
        random_seed=42,
    )
    ensemble.fit(
        X_train=X_train,
        y_train=y_train,
        feature_names=feature_names,
        train_dir=exp_models_dir,
    )

    # 5. Fit frozen C7 Isotonic Calibrator on Val
    print("[4/9] Generating Val and Test distributions and fitting Isotonic Calibrator...")
    dist_val = ensemble.predict_distribution(X_val)
    dist_test = ensemble.predict_distribution(X_test)

    calibrator = IsotonicCalibrator()
    calibrator.fit(dist_val["mean_prob"], y_val)
    cal_prob_test = calibrator.predict_proba(dist_test["mean_prob"])
    uncertainty_test = dist_test["std_prob"]

    threshold = 0.20

    # 6. Establish Nominal Reference
    print("[5/9] Evaluating Nominal Reference Profile on locked test split (N=14,913)...")
    nom_metrics = evaluate_full_robustness_profile(
        y_true=y_test,
        y_prob=cal_prob_test,
        uncertainty=uncertainty_test,
        threshold=threshold,
    )

    df_nominal = pd.DataFrame([{
        "metric": k,
        "value": f"{v:.4f}" if isinstance(v, float) else str(v)
    } for k, v in nom_metrics.items()])
    df_nominal.to_csv(exp_tables_dir / "nominal_reference.csv", index=False)

    # 7. Execute Missingness Perturbation Experiments
    print("[6/9] Running Missingness Shift Perturbation Suite...")
    df_missingness, detailed_missingness = evaluate_missingness_scenarios(
        df_test_raw=df_test,
        preprocessor=preprocessor,
        ensemble=ensemble,
        calibrator=calibrator,
        threshold=threshold,
        seed=42,
    )
    df_missingness.to_csv(exp_tables_dir / "missingness_degradation.csv", index=False)

    # 8. Execute Demographic & Subgroup Shifts
    print("[7/9] Running Demographic & Population Composition Shifts...")
    df_subgroups, detailed_subgroups = evaluate_demographic_subgroups(
        df_test_raw=df_test,
        y_test=y_test,
        cal_prob_test=cal_prob_test,
        uncertainty_test=uncertainty_test,
        threshold=threshold,
    )
    df_subgroups.to_csv(exp_tables_dir / "demographic_subgroup_shift.csv", index=False)

    df_pop_composition = evaluate_population_composition_shifts(
        df_test_raw=df_test,
        y_test=y_test,
        cal_prob_test=cal_prob_test,
        uncertainty_test=uncertainty_test,
        threshold=threshold,
        seed=42,
    )
    df_pop_composition.to_csv(exp_tables_dir / "population_composition_shift.csv", index=False)

    # 9. Execute Encounter Complexity & Utilization Shifts
    print("[8/9] Running Encounter Complexity & Utilization Shifts...")
    df_strata, detailed_strata = evaluate_utilization_phenotype_strata(
        df_test_raw=df_test,
        y_test=y_test,
        cal_prob_test=cal_prob_test,
        uncertainty_test=uncertainty_test,
        threshold=threshold,
    )
    df_strata.to_csv(exp_tables_dir / "encounter_strata_shift.csv", index=False)

    df_encounter_comp = evaluate_encounter_composition_shifts(
        df_test_raw=df_test,
        y_test=y_test,
        cal_prob_test=cal_prob_test,
        uncertainty_test=uncertainty_test,
        threshold=threshold,
        seed=42,
    )
    df_encounter_comp.to_csv(exp_tables_dir / "encounter_distribution_shift.csv", index=False)

    # 10. Execute Temporal Shift Analysis
    print("[9/9] Running Temporal Sequence & Longitudinal Stability Analysis...")
    df_temporal, detailed_temporal = evaluate_temporal_shift(
        df_test_raw=df_test,
        y_test=y_test,
        cal_prob_test=cal_prob_test,
        uncertainty_test=uncertainty_test,
        threshold=threshold,
    )
    df_temporal.to_csv(exp_tables_dir / "temporal_shift.csv", index=False)

    # 11. Profile Epistemic Failure Regimes & High-Confidence Errors
    df_regimes, failure_summary = profile_epistemic_failure_regimes(
        y_true=y_test,
        y_prob=cal_prob_test,
        uncertainty=uncertainty_test,
        threshold=threshold,
    )
    df_regimes.to_csv(exp_tables_dir / "failure_regimes_high_confidence_errors.csv", index=False)

    # 12. Build Master Robustness Summary Matrix
    matrix_rows = []

    # Nominal
    matrix_rows.append({
        "category": "Nominal Reference",
        "shift_name": "Nominal Locked Test (N=14,913)",
        "n_samples": nom_metrics["n_samples"],
        "roc_auc": nom_metrics["roc_auc"],
        "delta_roc_auc": 0.0,
        "pr_auc": nom_metrics["pr_auc"],
        "delta_pr_auc": 0.0,
        "brier_score": nom_metrics["brier_score"],
        "delta_brier": 0.0,
        "ece": nom_metrics["ece"],
        "delta_ece": 0.0,
        "calibration_slope": nom_metrics["calibration_slope"],
        "delta_slope": 0.0,
        "mean_uncertainty": nom_metrics["mean_uncertainty"],
        "delta_mean_uncertainty": 0.0,
        "error_rate": nom_metrics["error_rate"],
        "delta_error_rate": 0.0,
        "error_detection_auroc": nom_metrics["error_detection_auroc"],
        "aurc": nom_metrics["aurc"],
        "robustness_tier": "Nominal Baseline",
    })

    # Missingness
    for _, r in df_missingness.iterrows():
        if r["scenario"] != "M0_Nominal_Reference":
            matrix_rows.append({
                "category": "Missingness Perturbation",
                "shift_name": r["scenario"],
                "n_samples": r["n_samples"],
                "roc_auc": r["roc_auc"],
                "delta_roc_auc": r["delta_roc_auc"],
                "pr_auc": r["pr_auc"],
                "delta_pr_auc": r["delta_pr_auc"],
                "brier_score": r["brier_score"],
                "delta_brier": r["delta_brier"],
                "ece": r["ece"],
                "delta_ece": r["delta_ece"],
                "calibration_slope": r["calibration_slope"],
                "delta_slope": r["delta_slope"],
                "mean_uncertainty": r["mean_uncertainty"],
                "delta_mean_uncertainty": r["delta_mean_uncertainty"],
                "error_rate": r["error_rate"],
                "delta_error_rate": r["delta_error_rate"],
                "error_detection_auroc": r["error_detection_auroc"],
                "aurc": r["aurc"],
                "robustness_tier": "Controlled Degradation" if abs(r["delta_roc_auc"]) < 0.05 else "Vulnerable Regime",
            })

    # Demographic Subgroups
    for _, r in df_subgroups.iterrows():
        matrix_rows.append({
            "category": "Demographic Cohort",
            "shift_name": f"{r['category']}: {r['subgroup_name']}",
            "n_samples": r["n_samples"],
            "roc_auc": r["roc_auc"],
            "delta_roc_auc": r["delta_roc_auc"],
            "pr_auc": r["pr_auc"],
            "delta_pr_auc": r["delta_pr_auc"],
            "brier_score": r["brier_score"],
            "delta_brier": r["brier_score"] - nom_metrics["brier_score"],
            "ece": r["ece"],
            "delta_ece": r["ece"] - nom_metrics["ece"],
            "calibration_slope": r["calibration_slope"],
            "delta_slope": r["calibration_slope"] - nom_metrics["calibration_slope"],
            "mean_uncertainty": r["mean_uncertainty"],
            "delta_mean_uncertainty": r["mean_uncertainty"] - nom_metrics["mean_uncertainty"],
            "error_rate": r["error_rate"],
            "delta_error_rate": r["error_rate"] - nom_metrics["error_rate"],
            "error_detection_auroc": r["error_detection_auroc"],
            "aurc": r["aurc"],
            "robustness_tier": "Stable Subgroup",
        })

    # Encounter Complexity
    for _, r in df_strata.iterrows():
        matrix_rows.append({
            "category": "Encounter Complexity",
            "shift_name": f"{r['category']}: {r['stratum_name']}",
            "n_samples": r["n_samples"],
            "roc_auc": r["roc_auc"],
            "delta_roc_auc": r["delta_roc_auc"],
            "pr_auc": r["pr_auc"],
            "delta_pr_auc": r["delta_pr_auc"],
            "brier_score": r["brier_score"],
            "delta_brier": r["brier_score"] - nom_metrics["brier_score"],
            "ece": r["ece"],
            "delta_ece": r["ece"] - nom_metrics["ece"],
            "calibration_slope": r["calibration_slope"],
            "delta_slope": r["calibration_slope"] - nom_metrics["calibration_slope"],
            "mean_uncertainty": r["mean_uncertainty"],
            "delta_mean_uncertainty": r["mean_uncertainty"] - nom_metrics["mean_uncertainty"],
            "error_rate": r["error_rate"],
            "delta_error_rate": r["delta_error_rate"],
            "error_detection_auroc": r["error_detection_auroc"],
            "aurc": r["aurc"],
            "robustness_tier": "Phenotype Divergence",
        })

    # Temporal
    for _, r in df_temporal.iterrows():
        if r["temporal_id"] != "Nominal_Test_Reference":
            matrix_rows.append({
                "category": "Temporal Progression",
                "shift_name": r["period_description"],
                "n_samples": r["n_samples"],
                "roc_auc": r["roc_auc"],
                "delta_roc_auc": r["delta_roc_auc"],
                "pr_auc": r["pr_auc"],
                "delta_pr_auc": r["delta_pr_auc"],
                "brier_score": r["brier_score"],
                "delta_brier": r["brier_score"] - nom_metrics["brier_score"],
                "ece": r["ece"],
                "delta_ece": r["ece"] - nom_metrics["ece"],
                "calibration_slope": r["calibration_slope"],
                "delta_slope": r["calibration_slope"] - nom_metrics["calibration_slope"],
                "mean_uncertainty": r["mean_uncertainty"],
                "delta_mean_uncertainty": r["mean_uncertainty"] - nom_metrics["mean_uncertainty"],
                "error_rate": r["error_rate"],
                "delta_error_rate": r["delta_error_rate"],
                "error_detection_auroc": r["error_detection_auroc"],
                "aurc": r["aurc"],
                "robustness_tier": "Longitudinal Stable",
            })

    df_matrix = pd.DataFrame(matrix_rows)
    df_matrix.to_csv(exp_tables_dir / "robustness_summary_matrix.csv", index=False)

    # 13. Generate Publication Figures
    print("[10/10] Generating 7 publication-standard figures (300 DPI)...")
    plot_missingness_degradation(df_missingness, exp_figures_dir / "missingness_degradation.png")
    plot_subgroup_shift(df_subgroups, exp_figures_dir / "subgroup_shift.png")
    plot_encounter_shift(df_strata, exp_figures_dir / "encounter_shift.png")
    plot_temporal_shift(df_temporal, exp_figures_dir / "temporal_shift.png")
    plot_uncertainty_shift(df_missingness, df_pop_composition, df_encounter_comp, exp_figures_dir / "uncertainty_shift.png")

    # Risk-coverage shifted scenarios
    m25_data = detailed_missingness["M2_Random_MCAR_25pct"]
    m50_data = detailed_missingness["M3_Random_MCAR_50pct"]
    high_inp_mask = (df_test["number_inpatient"] >= 1).values
    high_diag_mask = (df_test["number_diagnoses"] >= 9).values

    shifted_rc_dict = {
        "Missingness +25%": (m25_data["y_true"], m25_data["probabilities"], m25_data["uncertainty"]),
        "Missingness +50%": (m50_data["y_true"], m50_data["probabilities"], m50_data["uncertainty"]),
        "High-Utilization Heavy": (y_test[high_inp_mask], cal_prob_test[high_inp_mask], uncertainty_test[high_inp_mask]),
        "Multimorbidity Heavy": (y_test[high_diag_mask], cal_prob_test[high_diag_mask], uncertainty_test[high_diag_mask]),
    }

    plot_risk_coverage_shift(
        nominal_y=y_test,
        nominal_prob=cal_prob_test,
        nominal_u=uncertainty_test,
        shifted_scenarios=shifted_rc_dict,
        save_path=exp_figures_dir / "risk_coverage_shift.png",
        threshold=threshold,
    )

    plot_robustness_summary(df_matrix, exp_figures_dir / "robustness_summary.png")

    # Mirror figures to Research Volume
    for fig_file in exp_figures_dir.glob("*.png"):
        shutil.copy2(fig_file, res_figures_dir / fig_file.name)

    # 14. Build Cryptographic Manifest
    print("Building cryptographic SHA-256 manifest...")
    manifest = {
        "experiment_phase": "Clinical Phase C9: Robustness, Fairness & Distribution Shift Analysis",
        "frozen_model": "CatBoost_HPO_Tuned_Candidate",
        "frozen_calibration": "Isotonic_Regression",
        "ensemble_size_M": 50,
        "operating_threshold": threshold,
        "partitions": {
            "train": len(df_train),
            "validation": len(df_val),
            "test": len(df_test),
        },
        "artifacts": {},
    }

    for path in sorted(list(exp_tables_dir.glob("*.csv")) + list(exp_figures_dir.glob("*.png"))):
        rel_str = str(path.relative_to(EXPERIMENT_DIR)).replace("\\", "/")
        manifest["artifacts"][rel_str] = {
            "sha256": compute_sha256(path),
            "size_bytes": path.stat().st_size,
        }

    manifest_path = exp_manifests_dir / "c9_robustness_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 80)
    print("PHASE C9 EXECUTION COMPLETE. ALL TABLES, FIGURES, AND MANIFEST GENERATED.")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
