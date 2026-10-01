"""
Master execution pipeline for Clinical Phase C7: Probability Calibration & Risk Reliability.
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

# Ensure project root in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.clinical.modeling.preprocessing import ClinicalPreprocessor
from catboost import CatBoostClassifier, Pool
from src.clinical.calibration.calibrator import PlattCalibrator, IsotonicCalibrator, BetaCalibrator
from src.clinical.calibration.metrics import (
    evaluate_calibration_metrics,
    compute_calibration_curve,
)
from src.clinical.calibration.dca import (
    compute_net_benefit,
    compute_operating_threshold_metrics,
)
from src.clinical.calibration.subgroup_calibration import (
    evaluate_subgroup_calibration,
)

EXPERIMENT_DIR = REPO_ROOT / "experiments" / "clinical" / "calibration"
RESEARCH_DIR = REPO_ROOT / "research" / "clinical" / "Volume_07_Probability_Calibration"


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def plot_reliability_diagrams(
    y_true_test: np.ndarray,
    prob_dict: Dict[str, np.ndarray],
    save_path: Path,
):
    """
    Generate multi-panel reliability diagram comparing Raw, Platt, Isotonic, and Beta calibration.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    axes = axes.flatten()

    colors = {
        "Raw CatBoost": "#2b5c8f",
        "Platt Scaling": "#d95f02",
        "Isotonic Regression": "#7570b3",
        "Beta Calibration": "#1b9e77",
    }

    for idx, (name, y_prob) in enumerate(prob_dict.items()):
        ax = axes[idx]
        prob_pred, prob_true, bin_counts = compute_calibration_curve(y_true_test, y_prob, n_bins=10)

        # Plot perfect calibration diagonal
        ax.plot([0, 1], [0, 1], "k--", alpha=0.7, label="Perfect Calibration")

        # Plot reliability curve
        color = colors.get(name, "#333333")
        ax.plot(prob_pred, prob_true, "s-", color=color, linewidth=2, markersize=7, label=name)

        # Calculate metrics for title/annotation
        m = evaluate_calibration_metrics(y_true_test, y_prob)
        ax.set_title(
            f"{name}\nECE={m['ece']:.4f} | Brier={m['brier_score']:.4f} | Slope={m['calibration_slope']:.3f}",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("Mean Predicted Probability", fontsize=10)
        ax.set_ylabel("Observed Readmission Rate", fontsize=10)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left", fontsize=9)

    plt.suptitle("Clinical 30-Day Readmission Reliability Curves (Locked Test Set, N=14,913)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_decision_curves(
    df_dca_raw: pd.DataFrame,
    df_dca_cal: pd.DataFrame,
    cal_name: str,
    save_path: Path,
):
    """
    Generate Decision Curve Analysis (DCA) Net Benefit plot across clinical operating thresholds.
    """
    plt.figure(figsize=(9, 6), dpi=300)

    # Plot Treat None
    plt.plot(df_dca_raw["threshold"], df_dca_raw["net_benefit_none"], "k-", linewidth=1.5, label="Treat None (Net Benefit = 0)")

    # Plot Treat All
    plt.plot(df_dca_raw["threshold"], df_dca_raw["net_benefit_all"], "gray", linestyle="--", linewidth=1.5, label="Treat All")

    # Plot Raw Model
    plt.plot(df_dca_raw["threshold"], df_dca_raw["net_benefit_model"], color="#2b5c8f", linestyle="-.", linewidth=2, label="Raw CatBoost Model")

    # Plot Calibrated Model
    plt.plot(df_dca_cal["threshold"], df_dca_cal["net_benefit_model"], color="#d95f02", linewidth=2.5, label=f"Calibrated Model ({cal_name})")

    plt.title("Decision Curve Analysis: 30-Day Readmission Risk Net Benefit", fontsize=12, fontweight="bold")
    plt.xlabel("Decision Threshold Probability (θ)", fontsize=11)
    plt.ylabel("Net Benefit", fontsize=11)
    plt.xlim([0.05, 0.40])
    plt.ylim([-0.02, 0.12])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=10)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def run_pipeline():
    print("=" * 80)
    print("FUSIONMEDAI: CLINICAL PHASE C7 PROBABILITY CALIBRATION PIPELINE")
    print("=" * 80)

    # 1. Setup output paths
    exp_tables_dir = EXPERIMENT_DIR / "tables"
    exp_figures_dir = EXPERIMENT_DIR / "figures"
    exp_manifests_dir = EXPERIMENT_DIR / "manifests"

    for d in [exp_tables_dir, exp_figures_dir, exp_manifests_dir]:
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

    # 4. Instantiate and fit frozen CatBoost HPO candidate
    print("[3/8] Instantiating frozen CatBoost HPO candidate model...")
    cb_train_dir = EXPERIMENT_DIR / "catboost_train"
    cb_train_dir.mkdir(parents=True, exist_ok=True)
    cb_model = CatBoostClassifier(
        depth=4,
        learning_rate=0.1383,
        iterations=350,
        l2_leaf_reg=2.911,
        subsample=0.655,
        random_seed=42,
        verbose=0,
        thread_count=-1,
        train_dir=str(cb_train_dir),
    )
    cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)

    # 5. Extract raw predictions & margins
    print("[4/8] Generating raw probability predictions and log-odds margins...")
    raw_prob_val = cb_model.predict_proba(X_val)[:, 1]
    raw_prob_test = cb_model.predict_proba(X_test)[:, 1]

    # Margin logits: ln(p / (1 - p))
    eps = 1e-12
    raw_margin_val = np.log(np.clip(raw_prob_val, eps, 1 - eps) / (1.0 - np.clip(raw_prob_val, eps, 1 - eps)))
    raw_margin_test = np.log(np.clip(raw_prob_test, eps, 1 - eps) / (1.0 - np.clip(raw_prob_test, eps, 1 - eps)))

    # 6. Fit calibrators strictly on Validation
    print("[5/8] Fitting candidate calibrators (Platt, Isotonic, Beta) strictly on Validation...")
    calibrators = {
        "Raw CatBoost": None,
        "Platt Scaling": PlattCalibrator().fit(raw_prob_val, y_val, logits_val=raw_margin_val),
        "Isotonic Regression": IsotonicCalibrator().fit(raw_prob_val, y_val),
        "Beta Calibration": BetaCalibrator().fit(raw_prob_val, y_val),
    }

    # 7. Evaluate on Validation & Test
    print("[6/8] Evaluating discrimination & calibration metrics across partitions...")
    val_prob_map = {}
    test_prob_map = {}
    comparison_rows = []

    for name, cal in calibrators.items():
        if cal is None:
            p_val = raw_prob_val
            p_test = raw_prob_test
        else:
            p_val = cal.predict_proba(raw_prob_val, logits=raw_margin_val)
            p_test = cal.predict_proba(raw_prob_test, logits=raw_margin_test)

        val_prob_map[name] = p_val
        test_prob_map[name] = p_test

        m_val = evaluate_calibration_metrics(y_val, p_val)
        m_test = evaluate_calibration_metrics(y_test, p_test)

        comparison_rows.append({
            "model": name,
            "val_log_loss": m_val["log_loss"],
            "val_brier": m_val["brier_score"],
            "val_ece": m_val["ece"],
            "val_mce": m_val["mce"],
            "val_intercept": m_val["calibration_intercept"],
            "val_slope": m_val["calibration_slope"],
            "val_roc_auc": m_val["roc_auc"],
            "val_pr_auc": m_val["pr_auc"],
            "test_log_loss": m_test["log_loss"],
            "test_brier": m_test["brier_score"],
            "test_ece": m_test["ece"],
            "test_mce": m_test["mce"],
            "test_intercept": m_test["calibration_intercept"],
            "test_slope": m_test["calibration_slope"],
            "test_roc_auc": m_test["roc_auc"],
            "test_pr_auc": m_test["pr_auc"],
        })

    df_comparison = pd.DataFrame(comparison_rows)
    df_comparison.to_csv(exp_tables_dir / "calibration_comparison.csv", index=False)
    print("Calibration Comparison Scoreboard:")
    print(df_comparison[["model", "val_log_loss", "val_ece", "test_log_loss", "test_ece", "test_slope", "test_roc_auc"]].to_string(index=False))

    # Determine optimal calibrator by Validation Log Loss
    best_cal_name = df_comparison.sort_values("val_log_loss").iloc[0]["model"]
    print(f"--> Optimal Calibrator selected by Validation Log Loss: {best_cal_name}")

    best_cal_test_prob = test_prob_map[best_cal_name]

    # 8. Subgroup Calibration Audit
    print("[7/8] Auditing subgroup calibration across Utilization, Gender, and Age cohorts...")
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    df_subgroup = evaluate_subgroup_calibration(
        X_test_df, y_test, raw_prob_test, best_cal_test_prob
    )
    df_subgroup.to_csv(exp_tables_dir / "subgroup_calibration.csv", index=False)

    # 9. Operating Threshold & DCA Sweeps
    print("[8/8] Performing post-calibration threshold sweep and Decision Curve Analysis...")
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    df_thresh = compute_operating_threshold_metrics(y_test, best_cal_test_prob, thresholds)
    df_thresh.to_csv(exp_tables_dir / "threshold_analysis.csv", index=False)

    dca_thresholds = np.linspace(0.01, 0.50, 100)
    df_dca_raw = compute_net_benefit(y_test, raw_prob_test, dca_thresholds)
    df_dca_cal = compute_net_benefit(y_test, best_cal_test_prob, dca_thresholds)
    df_dca_cal.to_csv(exp_tables_dir / "decision_curve_analysis.csv", index=False)

    # 10. Generate Figures
    rel_fig_path = exp_figures_dir / "reliability_diagrams.png"
    plot_reliability_diagrams(y_test, test_prob_map, rel_fig_path)

    dca_fig_path = exp_figures_dir / "decision_curve_analysis.png"
    plot_decision_curves(df_dca_raw, df_dca_cal, best_cal_name, dca_fig_path)

    # Mirror figures to research volume
    research_fig_dir = RESEARCH_DIR / "figures"
    research_fig_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(rel_fig_path, research_fig_dir / "reliability_diagrams.png")
    shutil.copy2(dca_fig_path, research_fig_dir / "decision_curve_analysis.png")

    # 11. Generate Cryptographic Manifest
    manifest_records = {}
    for p in sorted(EXPERIMENT_DIR.rglob("*")):
        if p.is_file() and p.suffix != ".json":
            manifest_records[str(p.relative_to(EXPERIMENT_DIR)).replace("\\", "/")] = {
                "sha256": compute_sha256(p),
                "size_bytes": p.stat().st_size,
            }

    manifest = {
        "experiment_phase": "Clinical Phase C7: Probability Calibration & Risk Reliability",
        "frozen_model": "CatBoost HPO Tuned (depth=4, lr=0.1383, iterations=350, l2=2.911, subsample=0.655, seed=42)",
        "feature_dimension": 119,
        "selected_calibrator": best_cal_name,
        "selection_metric": "Validation Log Loss (NLL)",
        "partitions": {
            "train": int(len(y_train)),
            "validation": int(len(y_val)),
            "test": int(len(y_test)),
        },
        "artifacts": manifest_records,
    }

    manifest_path = exp_manifests_dir / "c7_calibration_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[SUCCESS] Phase C7 Calibration Pipeline Complete!")
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    run_pipeline()
