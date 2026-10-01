"""
Master execution pipeline for Clinical Phase C8: Prediction Uncertainty Estimation.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Ensure project root in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.clinical.modeling.preprocessing import ClinicalPreprocessor
from src.clinical.calibration.calibrator import IsotonicCalibrator, BetaCalibrator
from src.clinical.uncertainty.bootstrap_ensemble import BootstrapCatBoostEnsemble
from src.clinical.uncertainty.evaluation import (
    evaluate_error_detection,
    evaluate_risk_coverage,
    evaluate_threshold_uncertainty_tiers,
    evaluate_subgroup_uncertainty,
    evaluate_convergence,
)

EXPERIMENT_DIR = REPO_ROOT / "experiments" / "clinical" / "uncertainty"
RESEARCH_DIR = REPO_ROOT / "research" / "clinical" / "Volume_08_Uncertainty"


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def plot_error_detection(
    uncertainty: np.ndarray,
    errors: np.ndarray,
    save_path: Path,
    error_auroc: float,
):
    """Generate 2-panel figure: uncertainty distribution for correct vs incorrect + error ROC curve."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Panel A: Uncertainty Distributions
    unc_correct = uncertainty[errors == 0]
    unc_incorrect = uncertainty[errors == 1]

    axes[0].hist(unc_correct, bins=40, density=True, alpha=0.6, color="#2b83ba", label=f"Correct (N={len(unc_correct):,})")
    axes[0].hist(unc_incorrect, bins=40, density=True, alpha=0.6, color="#d7191c", label=f"Incorrect (N={len(unc_incorrect):,})")
    axes[0].axvline(np.mean(unc_correct), color="#2b83ba", linestyle="--", linewidth=1.5, label=f"Mean Correct: {np.mean(unc_correct):.4f}")
    axes[0].axvline(np.mean(unc_incorrect), color="#d7191c", linestyle="--", linewidth=1.5, label=f"Mean Incorrect: {np.mean(unc_incorrect):.4f}")
    axes[0].set_title("Predictive Uncertainty by Classification Accuracy", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Predictive Standard Deviation (σ_p)", fontsize=10)
    axes[0].set_ylabel("Density", fontsize=10)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=8, loc="upper right")

    # Panel B: Error Detection ROC Curve
    fpr, tpr, _ = roc_curve(errors, uncertainty)
    axes[1].plot(fpr, tpr, color="#7b3294", linewidth=2.5, label=f"Uncertainty Detector (AUROC = {error_auroc:.4f})")
    axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.0, label="Random Guess (AUROC = 0.5000)")
    axes[1].set_title("Error Detection ROC Curve (Target: y_pred ≠ y_true)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("False Alarm Rate (Correct Cases Flagged as Uncertain)", fontsize=10)
    axes[1].set_ylabel("Error Detection Rate (True Errors Flagged)", fontsize=10)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_risk_coverage(
    df_rc: pd.DataFrame,
    summary: Dict[str, float],
    save_path: Path,
):
    """Plot Risk-Coverage curve comparing empirical rejection curve against random baseline."""
    plt.figure(figsize=(7, 5), dpi=300)

    # Empirical Rejection Curve
    plt.plot(
        df_rc["coverage"] * 100.0,
        df_rc["risk_error_rate"] * 100.0,
        marker="o",
        color="#1b7837",
        linewidth=2.5,
        label=f"Bootstrap Uncertainty (AURC = {summary['aurc']:.4f})",
    )

    # Random Rejection Baseline (Flat error rate)
    plt.axhline(
        summary["baseline_error_rate"] * 100.0,
        color="gray",
        linestyle="--",
        linewidth=1.2,
        label=f"Random Rejection (Baseline Risk = {summary['baseline_error_rate']*100:.2f}%)",
    )

    plt.title("Selective Classification: Risk-Coverage Curve (θ = 0.20)", fontsize=11, fontweight="bold")
    plt.xlabel("Coverage (% of Encounters Retained)", fontsize=10)
    plt.ylabel("Residual Risk / Error Rate (%)", fontsize=10)
    plt.xlim([10, 105])
    plt.ylim([0, max(df_rc["risk_error_rate"] * 100.0) * 1.3])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", fontsize=9)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_threshold_scatter(
    mean_probs: np.ndarray,
    uncertainty: np.ndarray,
    y_true: np.ndarray,
    save_path: Path,
    threshold: float = 0.20,
):
    """Plot scatter of predicted probability vs uncertainty, highlighting decision boundary and ambiguity zone."""
    plt.figure(figsize=(8, 6), dpi=300)

    unc_75th = float(np.percentile(uncertainty, 75.0))

    # Subsample for clear visualization (5,000 points)
    n_plot = min(5000, len(y_true))
    idx = np.random.RandomState(42).choice(len(y_true), size=n_plot, replace=False)

    p_sub = mean_probs[idx]
    u_sub = uncertainty[idx]
    y_sub = y_true[idx]

    plt.scatter(
        p_sub[y_sub == 0],
        u_sub[y_sub == 0],
        alpha=0.25,
        color="#4575b4",
        s=12,
        label="Non-Readmitted (y = 0)",
    )
    plt.scatter(
        p_sub[y_sub == 1],
        u_sub[y_sub == 1],
        alpha=0.45,
        color="#d73027",
        s=18,
        label="Readmitted (y = 1)",
    )

    # Decision threshold line
    plt.axvline(threshold, color="#313695", linestyle="--", linewidth=2.0, label=f"Decision Threshold θ = {threshold:.2f}")
    
    # Uncertainty cutoff line
    plt.axhline(unc_75th, color="#a50026", linestyle=":", linewidth=1.5, label=f"75th Percentile Uncertainty = {unc_75th:.4f}")

    # Ambiguity shading around threshold
    plt.axvspan(threshold - 0.03, threshold + 0.03, color="orange", alpha=0.15, label="Decision Ambiguity Zone [θ ± 0.03]")

    plt.title("Risk Probability vs. Predictive Uncertainty Scatter", fontsize=11, fontweight="bold")
    plt.xlabel("Ensemble Mean Predicted Probability", fontsize=10)
    plt.ylabel("Predictive Standard Deviation (σ_p)", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=8)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_convergence(
    df_conv: pd.DataFrame,
    save_path: Path,
):
    """Plot convergence of ranking correlation and Error-detection AUROC vs ensemble size."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Panel A: Ranking Stability
    axes[0].plot(
        df_conv["ensemble_size_M"],
        df_conv["spearman_corr_std"],
        marker="s",
        color="#2c7bb6",
        linewidth=2.0,
        label="Uncertainty Ranking Corr (vs M=50)",
    )
    axes[0].plot(
        df_conv["ensemble_size_M"],
        df_conv["spearman_corr_mean"],
        marker="o",
        color="#008837",
        linewidth=2.0,
        label="Mean Probability Corr (vs M=50)",
    )
    axes[0].set_title("Ensemble Ranking Stability (Spearman ρ)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Number of Bootstrap Models (M)", fontsize=10)
    axes[0].set_ylabel("Spearman Rank Correlation", fontsize=10)
    axes[0].set_ylim([0.70, 1.02])
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=9, loc="lower right")

    # Panel B: Error Detection AUROC
    axes[1].plot(
        df_conv["ensemble_size_M"],
        df_conv["error_detection_auroc"],
        marker="d",
        color="#7b3294",
        linewidth=2.0,
        label="Error Detection AUROC (θ = 0.20)",
    )
    axes[1].set_title("Error Detection Utility vs. Ensemble Size", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Number of Bootstrap Models (M)", fontsize=10)
    axes[1].set_ylabel("AUROC", fontsize=10)
    axes[1].set_ylim([0.48, 0.70])
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def run_pipeline():
    print("=" * 80)
    print("FUSIONMEDAI: CLINICAL PHASE C8 PREDICTION UNCERTAINTY ESTIMATION PIPELINE")
    print("=" * 80)

    # 1. Setup output paths
    exp_tables_dir = EXPERIMENT_DIR / "tables"
    exp_figures_dir = EXPERIMENT_DIR / "figures"
    exp_manifests_dir = EXPERIMENT_DIR / "manifests"
    exp_models_dir = EXPERIMENT_DIR / "models"

    for d in [exp_tables_dir, exp_figures_dir, exp_manifests_dir, exp_models_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 2. Load dataset partitions
    data_dir = REPO_ROOT / "datasets" / "clinical" / "processed" / "splits"
    print(f"[1/8] Loading frozen C2 splits from {data_dir}...")
    df_train = pd.read_csv(data_dir / "train.csv")
    df_val = pd.read_csv(data_dir / "val.csv")
    df_test = pd.read_csv(data_dir / "test.csv")

    # 3. Fit locked Preprocessor on train
    print("[2/8] Fitting locked ClinicalPreprocessor (D=119)...")
    preprocessor = ClinicalPreprocessor(scale_numerical=True)
    preprocessor.fit(df_train)
    X_train, y_train, _ = preprocessor.transform(df_train)
    X_val, y_val, _ = preprocessor.transform(df_val)
    X_test, y_test, _ = preprocessor.transform(df_test)

    feature_names = list(preprocessor.feature_names_)
    assert len(feature_names) == 119, f"Expected 119 features, got {len(feature_names)}"

    # 4. Train Bootstrap Ensemble (M=50 models)
    n_estimators = 50
    print(f"[3/8] Training Bootstrap CatBoost Ensemble (M={n_estimators} members)...")
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

    # 5. Stochastic Inference on Val and Test
    print("[4/8] Generating stochastic prediction distributions for Val and Test...")
    dist_val = ensemble.predict_distribution(X_val)
    dist_test = ensemble.predict_distribution(X_test)

    # Load and apply frozen C7 calibration mapping (Option A)
    print("[5/8] Applying C7 validation-fitted calibration mapping...")
    calibrator = IsotonicCalibrator()
    calibrator.fit(dist_val["mean_prob"], y_val)
    cal_prob_test = calibrator.predict_proba(dist_test["mean_prob"])

    # 6. Evaluate Error Detection, Risk Coverage, Thresholds, and Subgroups
    print("[6/8] Evaluating error detection, risk-coverage, tiers, and subgroups...")
    threshold = 0.20
    errors_test = ((dist_test["mean_prob"] >= threshold).astype(int) != y_test).astype(int)

    # Error detection
    err_det_raw = evaluate_error_detection(y_test, dist_test["mean_prob"], dist_test["std_prob"], threshold=threshold)
    err_det_cal = evaluate_error_detection(y_test, cal_prob_test, dist_test["std_prob"], threshold=threshold)

    # Risk-coverage
    df_rc, rc_summary = evaluate_risk_coverage(y_test, dist_test["mean_prob"], dist_test["std_prob"], threshold=threshold)

    # Threshold tiers
    df_tiers = evaluate_threshold_uncertainty_tiers(y_test, dist_test["mean_prob"], dist_test["std_prob"], threshold=threshold)

    # Subgroups
    df_subgroups = evaluate_subgroup_uncertainty(df_test, y_test, dist_test["mean_prob"], dist_test["std_prob"], threshold=threshold)

    # Convergence analysis
    df_conv = evaluate_convergence(dist_test["all_probs"], y_test, subset_sizes=[5, 10, 20, 30, 40, 50], threshold=threshold)

    # Core summary table
    summary_data = [{
        "metric": "Total Test Encounters",
        "value": f"{len(y_test):,}",
    }, {
        "metric": "Ensemble Size (M)",
        "value": f"{n_estimators}",
    }, {
        "metric": "Mean Ensemble Probability",
        "value": f"{np.mean(dist_test['mean_prob']):.4f}",
    }, {
        "metric": "Mean Predictive Uncertainty (std)",
        "value": f"{np.mean(dist_test['std_prob']):.4f}",
    }, {
        "metric": "Median Predictive Uncertainty (std)",
        "value": f"{np.median(dist_test['std_prob']):.4f}",
    }, {
        "metric": "IQR Predictive Uncertainty",
        "value": f"[{np.percentile(dist_test['std_prob'], 25):.4f}, {np.percentile(dist_test['std_prob'], 75):.4f}]",
    }, {
        "metric": "Error Detection AUROC (θ=0.20)",
        "value": f"{err_det_raw['error_auroc']:.4f}",
    }, {
        "metric": "Error Detection AUPRC (θ=0.20)",
        "value": f"{err_det_raw['error_auprc']:.4f}",
    }, {
        "metric": "Risk-Coverage AURC",
        "value": f"{rc_summary['aurc']:.4f}",
    }, {
        "metric": "Excess AURC (E-AURC)",
        "value": f"{rc_summary['e_aurc']:.4f}",
    }, {
        "metric": "Error Rate at 80% Coverage",
        "value": f"{rc_summary['error_at_80pct_coverage']*100:.2f}%",
    }, {
        "metric": "Baseline Error Rate (100% Coverage)",
        "value": f"{rc_summary['baseline_error_rate']*100:.2f}%",
    }]
    df_summary = pd.DataFrame(summary_data)

    # 7. Select 7 Representative Patient Cases connecting C6 SHAP + C7 Cal + C8 Unc
    print("[7/8] Extracting 7 representative patient cases...")
    
    # Find candidates
    u = dist_test["std_prob"]
    p = dist_test["mean_prob"]
    u_high = np.percentile(u, 85)
    u_low = np.percentile(u, 15)

    cases = []
    # Case 1: High Risk / Low Uncertainty (True Positive)
    c1_cand = np.where((p >= 0.25) & (u <= u_low) & (y_test == 1))[0]
    idx1 = int(c1_cand[0]) if len(c1_cand) > 0 else int(np.argmax(p))

    # Case 2: High Risk / High Uncertainty (True Positive)
    c2_cand = np.where((p >= 0.25) & (u >= u_high) & (y_test == 1))[0]
    idx2 = int(c2_cand[0]) if len(c2_cand) > 0 else int(np.argmax(u))

    # Case 3: Low Risk / Low Uncertainty (True Negative)
    c3_cand = np.where((p <= 0.08) & (u <= u_low) & (y_test == 0))[0]
    idx3 = int(c3_cand[0]) if len(c3_cand) > 0 else int(np.argmin(p))

    # Case 4: Low Risk / High Uncertainty (True Negative)
    c4_cand = np.where((p <= 0.12) & (u >= u_high) & (y_test == 0))[0]
    idx4 = int(c4_cand[0]) if len(c4_cand) > 0 else int(np.argmax(u))

    # Case 5: Near Threshold Ambiguity (|p - 0.20| <= 0.02, High Unc)
    c5_cand = np.where((np.abs(p - 0.20) <= 0.02) & (u >= u_high))[0]
    idx5 = int(c5_cand[0]) if len(c5_cand) > 0 else int(np.argmin(np.abs(p - 0.20)))

    # Case 6: False Positive with High Uncertainty
    c6_cand = np.where((p >= 0.20) & (y_test == 0) & (u >= u_high))[0]
    idx6 = int(c6_cand[0]) if len(c6_cand) > 0 else idx2

    # Case 7: False Negative with High Uncertainty
    c7_cand = np.where((p < 0.15) & (y_test == 1) & (u >= u_high))[0]
    idx7 = int(c7_cand[0]) if len(c7_cand) > 0 else idx4

    selected_indices = [
        ("Case 1: High Risk / Low Uncertainty", idx1, "Stable High-Risk Candidate"),
        ("Case 2: High Risk / High Uncertainty", idx2, "High-Risk with Data Sparsity"),
        ("Case 3: Low Risk / Low Uncertainty", idx3, "Stable Low-Risk Baseline"),
        ("Case 4: Low Risk / High Uncertainty", idx4, "Low-Risk Review Candidate"),
        ("Case 5: Near-Threshold Ambiguity", idx5, "Decision Boundary Ambiguity"),
        ("Case 6: False Positive (High Uncertainty)", idx6, "Uncertain High-Alert Case"),
        ("Case 7: False Negative (High Uncertainty)", idx7, "Uncertain Missed Case"),
    ]

    case_rows = []
    for case_label, idx_case, interp in selected_indices:
        raw_p = float(dist_test["mean_prob"][idx_case])
        cal_p = float(cal_prob_test[idx_case])
        std_u = float(dist_test["std_prob"][idx_case])
        q025 = float(dist_test["q025"][idx_case])
        q975 = float(dist_test["q975"][idx_case])
        y_val_case = int(y_test[idx_case])
        pred_label = int(raw_p >= threshold)

        # Primary features from metadata
        num_inp = df_test.iloc[idx_case]["number_inpatient"] if "number_inpatient" in df_test.columns else "N/A"
        age_str = df_test.iloc[idx_case]["age"] if "age" in df_test.columns else "N/A"
        diag1 = df_test.iloc[idx_case]["diag_1"] if "diag_1" in df_test.columns else "N/A"

        case_rows.append({
            "case_id": case_label,
            "encounter_index": idx_case,
            "raw_mean_prob": raw_p,
            "calibrated_prob": cal_p,
            "uncertainty_std": std_u,
            "pi_95_lower": q025,
            "pi_95_upper": q975,
            "model_pred_020": pred_label,
            "true_outcome": y_val_case,
            "prior_inpatient": num_inp,
            "age_group": age_str,
            "primary_diag": diag1,
            "interpretation": interp,
        })

    df_cases = pd.DataFrame(case_rows)

    # 8. Export Tables & Generate Figures
    print("[8/8] Exporting tables and plotting publication figures...")
    df_summary.to_csv(exp_tables_dir / "uncertainty_summary.csv", index=False)
    df_rc.to_csv(exp_tables_dir / "risk_coverage.csv", index=False)
    df_tiers.to_csv(exp_tables_dir / "threshold_uncertainty_tiers.csv", index=False)
    df_subgroups.to_csv(exp_tables_dir / "subgroup_uncertainty.csv", index=False)
    df_conv.to_csv(exp_tables_dir / "convergence_analysis.csv", index=False)
    df_cases.to_csv(exp_tables_dir / "local_uncertainty_cases.csv", index=False)

    # Figures
    fig1 = exp_figures_dir / "error_detection_distributions.png"
    plot_error_detection(dist_test["std_prob"], errors_test, fig1, err_det_raw["error_auroc"])

    fig2 = exp_figures_dir / "risk_coverage_curve.png"
    plot_risk_coverage(df_rc, rc_summary, fig2)

    fig3 = exp_figures_dir / "threshold_uncertainty_scatter.png"
    plot_threshold_scatter(dist_test["mean_prob"], dist_test["std_prob"], y_test, fig3, threshold=threshold)

    fig4 = exp_figures_dir / "convergence_analysis.png"
    plot_convergence(df_conv, fig4)

    # Mirror figures to Research Volume
    res_fig_dir = RESEARCH_DIR / "figures"
    res_fig_dir.mkdir(parents=True, exist_ok=True)
    for fig_path in [fig1, fig2, fig3, fig4]:
        shutil.copy2(fig_path, res_fig_dir / fig_path.name)

    # 9. Build Cryptographic Manifest
    manifest_dict = {
        "experiment_phase": "Clinical Phase C8: Prediction Uncertainty Estimation",
        "ensemble_type": "BootstrapCatBoostEnsemble",
        "ensemble_size_M": n_estimators,
        "frozen_model": "CatBoost HPO Tuned (depth=4, lr=0.1383, iterations=350, l2=2.911, subsample=0.655, seed=42)",
        "frozen_c7_calibrator": "Isotonic Regression",
        "feature_dimension": 119,
        "partitions": {
            "train": int(len(X_train)),
            "validation": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "metrics_summary": {
            "mean_uncertainty_test": float(np.mean(dist_test["std_prob"])),
            "median_uncertainty_test": float(np.median(dist_test["std_prob"])),
            "error_detection_auroc_020": float(err_det_raw["error_auroc"]),
            "risk_coverage_aurc": float(rc_summary["aurc"]),
            "excess_aurc": float(rc_summary["e_aurc"]),
        },
        "artifacts": {},
    }

    # Collect all table and figure files
    all_artifact_files = (
        list(exp_tables_dir.glob("*.csv")) +
        list(exp_figures_dir.glob("*.png"))
    )

    for p in sorted(all_artifact_files):
        rel_str = str(p.relative_to(EXPERIMENT_DIR)).replace("\\", "/")
        manifest_dict["artifacts"][rel_str] = {
            "sha256": compute_sha256(p),
            "size_bytes": p.stat().st_size,
        }

    manifest_path = exp_manifests_dir / "c8_uncertainty_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_dict, f, indent=2)

    print(f"\n[COMPLETE] Phase C8 Pipeline finished successfully.")
    print(f"  -> Tables: {len(list(exp_tables_dir.glob('*.csv')))} CSVs generated.")
    print(f"  -> Figures: {len(list(exp_figures_dir.glob('*.png')))} PNGs generated and mirrored.")
    print(f"  -> Manifest: {manifest_path} ({len(manifest_dict['artifacts'])} artifacts tracked).")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
