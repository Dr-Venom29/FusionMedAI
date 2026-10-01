"""Master Execution Orchestrator for Phase C6 Clinical Model Explainability."""

import sys
import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.clinical.explainability.shap_analysis import ClinicalSHAPExplainer, FROZEN_CATBOOST_HPO_CONFIG
from src.clinical.explainability.global_explanation import (
    compute_global_importance,
    plot_global_bar,
    plot_shap_summary,
    plot_dependence,
)
from src.clinical.explainability.group_analysis import (
    get_clinical_feature_group,
    compute_group_importance,
    plot_group_importance,
)
from src.clinical.explainability.local_explanation import (
    generate_representative_local_cases,
    compute_error_case_attribution,
)
from src.clinical.benchmarking.runtime import get_runtime_output_root


def sha256_file(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def run_c6_pipeline():
    print("=" * 80)
    print("FusionMedAI: Phase C6 Clinical Explainability Execution")
    print("=" * 80)

    # 1. Output directory structure: experiments/clinical/explainability/
    exp_root = get_runtime_output_root(REPO_ROOT, "explainability")
    dir_global = exp_root / "global"
    dir_local = exp_root / "local"
    dir_subgroup = exp_root / "subgroup"
    dir_error = exp_root / "error_analysis"
    dir_figures = exp_root / "figures"
    dir_dependence = dir_figures / "dependence"
    dir_manifests = exp_root / "manifests"

    for d in [dir_global, dir_local, dir_subgroup, dir_error, dir_figures, dir_dependence, dir_manifests]:
        d.mkdir(parents=True, exist_ok=True)

    # Research directory structure: research/clinical/Volume_06_Explainability/
    research_root = REPO_ROOT / "research" / "clinical" / "Volume_06_Explainability"
    res_figures = research_root / "figures"
    res_dep = res_figures / "dependence"
    res_figures.mkdir(parents=True, exist_ok=True)
    res_dep.mkdir(parents=True, exist_ok=True)

    # 2. Load dataset splits
    splits_dir = REPO_ROOT / "datasets" / "clinical" / "processed" / "splits"
    print(f"[*] Loading frozen canonical splits from: {splits_dir}")
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")
    print(f"    Train: {train_df.shape} | Val: {val_df.shape} | Test: {test_df.shape}")

    # 3. Fit frozen CatBoost HPO candidate
    print("[*] Initializing and fitting frozen CatBoost HPO candidate (Trial 2)...")
    explainer = ClinicalSHAPExplainer(config=FROZEN_CATBOOST_HPO_CONFIG)
    explainer.fit_and_prepare(train_df, val_df)
    feature_names = explainer.feature_names
    print(f"[OK] Model prepared. Feature count: {len(feature_names)}")

    # 4. Generate SHAP values for Validation and Test splits
    print("[*] Computing TreeSHAP values for Validation partition (N=14,911)...")
    val_shap, val_base, X_val, val_probs, y_val = explainer.compute_shap_values(val_df)

    print("[*] Computing TreeSHAP values for Locked Test partition (N=14,913)...")
    test_shap, test_base, X_test, test_probs, y_test = explainer.compute_shap_values(test_df)
    print(f"[OK] Test SHAP matrix shape: {test_shap.shape} (Base log-odds: {test_base:.4f})")

    # 5. Global Feature Importance
    print("[*] Calculating global feature importance and directionalities...")
    global_df = compute_global_importance(test_shap, val_shap, X_test, feature_names)
    
    global_csv_path = dir_global / "global_feature_importance.csv"
    global_df.to_csv(global_csv_path, index=False)
    global_df.to_csv(research_root / "02_Global_Feature_Importance.csv", index=False)
    print(f"[OK] Saved global feature importance: {global_csv_path}")

    # 6. Clinical Feature Group Importance
    print("[*] Aggregating feature groups...")
    group_df = compute_group_importance(global_df)
    
    group_csv_path = dir_global / "feature_group_importance.csv"
    group_df.to_csv(group_csv_path, index=False)
    group_df.to_csv(research_root / "03_Feature_Group_Importance.csv", index=False)
    print(f"[OK] Saved feature group importance: {group_csv_path}")

    # 7. Local Case Explanations
    print("[*] Generating representative local case explanations...")
    local_df = generate_representative_local_cases(test_shap, test_probs, y_test, feature_names, test_base, threshold=0.20)
    
    local_csv_path = dir_local / "local_case_explanations.csv"
    local_df.to_csv(local_csv_path, index=False)
    local_df.to_csv(research_root / "05_Local_Explanations.csv", index=False)
    print(f"[OK] Saved local case explanations: {local_csv_path}")

    # 8. Error-Focused SHAP Analysis (FP vs FN)
    print("[*] Computing error-focused SHAP attributions (False Positives vs False Negatives)...")
    fp_df, fn_df = compute_error_case_attribution(test_shap, test_probs, y_test, feature_names, threshold=0.20)
    
    fp_csv_path = dir_error / "false_positive_shap_attributions.csv"
    fn_csv_path = dir_error / "false_negative_shap_attributions.csv"
    fp_df.to_csv(fp_csv_path, index=False)
    fn_df.to_csv(fn_csv_path, index=False)
    fp_df.to_csv(research_root / "07_False_Positive_Attributions.csv", index=False)
    fn_df.to_csv(research_root / "07_False_Negative_Attributions.csv", index=False)
    print(f"[OK] Saved error attribution tables: {fp_csv_path}, {fn_csv_path}")

    # 9. Subgroup SHAP Analysis (Age, Race, Gender, Prior Inpatient)
    print("[*] Computing demographic and clinical subgroup SHAP stability...")
    subgroup_records = []
    
    # Stratify by Prior Inpatient History
    prior_inpatient_raw = test_df["number_inpatient"].values
    inpatient_0_mask = prior_inpatient_raw == 0
    inpatient_ge1_mask = prior_inpatient_raw >= 1

    shap_inpatient_0 = np.mean(np.abs(test_shap[inpatient_0_mask]), axis=0)
    shap_inpatient_ge1 = np.mean(np.abs(test_shap[inpatient_ge1_mask]), axis=0)

    # Stratify by Gender
    if "gender_Female" in feature_names and "gender_Male" in feature_names:
        idx_f = feature_names.index("gender_Female")
        idx_m = feature_names.index("gender_Male")
        female_mask = X_test[:, idx_f] > 0.5
        male_mask = X_test[:, idx_m] > 0.5
        shap_female = np.mean(np.abs(test_shap[female_mask]), axis=0)
        shap_male = np.mean(np.abs(test_shap[male_mask]), axis=0)
    else:
        shap_female = shap_male = np.mean(np.abs(test_shap), axis=0)

    for i, fname in enumerate(feature_names):
        subgroup_records.append({
            "feature": fname,
            "feature_group": get_clinical_feature_group(fname),
            "mean_abs_shap_overall": float(global_df.loc[global_df["feature"] == fname, "mean_abs_shap_test"].values[0]),
            "mean_abs_shap_prior_inpatient_0": float(shap_inpatient_0[i]),
            "mean_abs_shap_prior_inpatient_ge1": float(shap_inpatient_ge1[i]),
            "mean_abs_shap_female": float(shap_female[i]),
            "mean_abs_shap_male": float(shap_male[i]),
        })

    subgroup_df = pd.DataFrame(subgroup_records).sort_values("mean_abs_shap_overall", ascending=False).reset_index(drop=True)
    subgroup_csv_path = dir_subgroup / "subgroup_feature_importance.csv"
    subgroup_df.to_csv(subgroup_csv_path, index=False)
    subgroup_df.to_csv(research_root / "08_Subgroup_Feature_Importance.csv", index=False)
    print(f"[OK] Saved subgroup importance table: {subgroup_csv_path}")

    # 10. Validation vs Test Stability Analysis
    print("[*] Computing validation vs test stability metrics...")
    spearman_rho, spearman_pval = spearmanr(global_df["mean_abs_shap_val"], global_df["mean_abs_shap_test"])
    
    top10_test = set(global_df.nsmallest(10, "rank_test")["feature"])
    top10_val = set(global_df.sort_values("mean_abs_shap_val", ascending=False).head(10)["feature"])
    top10_overlap = len(top10_test.intersection(top10_val)) / 10.0

    top20_test = set(global_df.nsmallest(20, "rank_test")["feature"])
    top20_val = set(global_df.sort_values("mean_abs_shap_val", ascending=False).head(20)["feature"])
    top20_overlap = len(top20_test.intersection(top20_val)) / 20.0

    stability_df = global_df[["rank_test", "feature", "feature_group", "mean_abs_shap_test", "mean_abs_shap_val"]].copy()
    stability_df["rank_val"] = stability_df["mean_abs_shap_val"].rank(ascending=False).astype(int)
    stability_df["rank_shift"] = stability_df["rank_val"] - stability_df["rank_test"]

    stability_csv_path = dir_global / "shap_stability.csv"
    stability_df.to_csv(stability_csv_path, index=False)
    stability_df.to_csv(research_root / "04_SHAP_Stability.csv", index=False)
    print(f"[OK] Stability Metrics: Spearman Rho = {spearman_rho:.4f} (p={spearman_pval:.2e}), Top-10 Overlap = {top10_overlap*100:.0f}%, Top-20 Overlap = {top20_overlap*100:.0f}%")

    # 11. Dominant-Feature Analysis
    top5_share = float(global_df.head(5)["relative_contribution_pct"].sum())
    top10_share = float(global_df.head(10)["relative_contribution_pct"].sum())
    top20_share = float(global_df.head(20)["relative_contribution_pct"].sum())
    print(f"[OK] Cumulative Feature Share: Top 5 = {top5_share:.2f}%, Top 10 = {top10_share:.2f}%, Top 20 = {top20_share:.2f}%")

    # 12. Generate Visualizations
    print("[*] Generating publication-quality figures...")
    bar_path = dir_figures / "shap_bar.png"
    plot_global_bar(global_df, bar_path, top_n=20)
    plot_global_bar(global_df, res_figures / "shap_bar.png", top_n=20)

    summary_path = dir_figures / "shap_summary.png"
    plot_shap_summary(test_shap, X_test, feature_names, global_df, summary_path, top_n=20, sample_size=2000)
    plot_shap_summary(test_shap, X_test, feature_names, global_df, res_figures / "shap_summary.png", top_n=20, sample_size=2000)

    group_bar_path = dir_figures / "shap_group_importance.png"
    plot_group_importance(group_df, group_bar_path)
    plot_group_importance(group_df, res_figures / "shap_group_importance.png")

    # Stability scatter plot
    plt.figure(figsize=(8, 8), dpi=300)
    plt.scatter(global_df["mean_abs_shap_val"], global_df["mean_abs_shap_test"], color="#1f77b4", edgecolor="black", s=45, alpha=0.75)
    max_val = max(global_df["mean_abs_shap_val"].max(), global_df["mean_abs_shap_test"].max()) * 1.05
    plt.plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label=f"Identity (Spearman rho = {spearman_rho:.4f})")
    for idx, row in global_df.head(7).iterrows():
        plt.annotate(row["feature"], (row["mean_abs_shap_val"], row["mean_abs_shap_test"]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8, fontweight="bold")
    plt.xlabel("Validation Mean |SHAP|", fontsize=12, fontweight="bold")
    plt.ylabel("Test Mean |SHAP|", fontsize=12, fontweight="bold")
    plt.title("SHAP Feature Importance Stability: Validation vs. Test Set", fontsize=13, fontweight="bold", pad=15)
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(dir_figures / "shap_stability.png")
    plt.savefig(res_figures / "shap_stability.png")
    plt.close()

    # Dependence plots
    key_continuous = ["number_inpatient", "time_in_hospital", "num_medications", "number_diagnoses", "num_lab_procedures"]
    for feat in key_continuous:
        plot_dependence(feat, test_shap, X_test, test_probs, feature_names, dir_dependence / f"dep_{feat}.png")
        plot_dependence(feat, test_shap, X_test, test_probs, feature_names, res_dep / f"dep_{feat}.png")

    print("[OK] All figures generated.")

    # 13. Cryptographic Manifest
    manifest_files = [
        global_csv_path,
        group_csv_path,
        local_csv_path,
        fp_csv_path,
        fn_csv_path,
        subgroup_csv_path,
        stability_csv_path,
        bar_path,
        summary_path,
        group_bar_path,
        dir_figures / "shap_stability.png",
    ] + list(dir_dependence.glob("*.png"))

    manifest_artifacts = {}
    for fpath in sorted(manifest_files):
        try:
            rel = fpath.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            rel = fpath.as_posix()
        manifest_artifacts[rel] = {
            "size_bytes": fpath.stat().st_size,
            "sha256": sha256_file(fpath),
        }

    manifest_data = {
        "provenance": {
            "phase": "C6_Clinical_Model_Explainability",
            "model_family": "CatBoostClassifier",
            "model_identifier": "catboost_hpo_trial_2",
            "hyperparameters": FROZEN_CATBOOST_HPO_CONFIG,
            "feature_dimension": len(feature_names),
            "random_seed": 42,
            "train_encounters": len(train_df),
            "val_encounters": len(val_df),
            "test_encounters": len(test_df),
            "spearman_stability_rho": round(float(spearman_rho), 4),
            "top10_overlap_pct": round(float(top10_overlap * 100), 1),
            "top20_overlap_pct": round(float(top20_overlap * 100), 1),
            "top5_dominant_contribution_pct": round(float(top5_share), 2),
            "top10_dominant_contribution_pct": round(float(top10_share), 2),
            "top20_dominant_contribution_pct": round(float(top20_share), 2),
        },
        "artifacts": manifest_artifacts,
    }

    manifest_path = dir_manifests / "c6_explainability_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
    with open(research_root / "c6_manifest.json", "w") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"[OK] Cryptographic manifest written: {manifest_path}")
    print("=" * 80)
    print("Phase C6 Clinical Explainability Execution COMPLETE.")
    print("=" * 80)


if __name__ == "__main__":
    run_c6_pipeline()
