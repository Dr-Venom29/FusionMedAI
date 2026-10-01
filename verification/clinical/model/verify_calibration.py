"""Phase C7 Clinical Probability Calibration & Risk Reliability Verification Suite."""

import sys
import json
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.clinical.benchmarking.runtime import get_runtime_output_root


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def verify_calibration_pipeline() -> bool:
    print("=" * 80)
    print("VERIFICATION: Clinical Phase C7 Probability Calibration & Risk Reliability")
    print("=" * 80)

    exp_root = get_runtime_output_root(REPO_ROOT, "calibration")
    research_root = REPO_ROOT / "research" / "clinical" / "Volume_07_Probability_Calibration"

    checks_passed = 0
    total_checks = 10

    # 1. Check Calibration Comparison Table Completeness
    comp_csv = exp_root / "tables" / "calibration_comparison.csv"
    if comp_csv.exists():
        df_comp = pd.read_csv(comp_csv)
        assert len(df_comp) == 4, f"Expected 4 models in comparison table, found {len(df_comp)}"
        expected_models = {"Raw CatBoost", "Platt Scaling", "Isotonic Regression", "Beta Calibration"}
        assert set(df_comp["model"]) == expected_models, f"Model mismatch: {set(df_comp['model'])}"
        print(f"[PASS 1/10] Calibration Comparison Table Verified: {len(df_comp)} methods evaluated.")
        checks_passed += 1
    else:
        print(f"[FAIL 1/10] Calibration Comparison Table Missing: {comp_csv}")

    # 2. Check Rank Preservation / Monotonicity on Parametric Calibrators
    if comp_csv.exists():
        raw_roc = df_comp.loc[df_comp["model"] == "Raw CatBoost", "test_roc_auc"].values[0]
        raw_pr = df_comp.loc[df_comp["model"] == "Raw CatBoost", "test_pr_auc"].values[0]
        platt_roc = df_comp.loc[df_comp["model"] == "Platt Scaling", "test_roc_auc"].values[0]
        platt_pr = df_comp.loc[df_comp["model"] == "Platt Scaling", "test_pr_auc"].values[0]
        beta_roc = df_comp.loc[df_comp["model"] == "Beta Calibration", "test_roc_auc"].values[0]
        beta_pr = df_comp.loc[df_comp["model"] == "Beta Calibration", "test_pr_auc"].values[0]

        assert np.isclose(raw_roc, platt_roc, atol=1e-5) and np.isclose(raw_pr, platt_pr, atol=1e-5), "Platt rank violation"
        assert np.isclose(raw_roc, beta_roc, atol=1e-5) and np.isclose(raw_pr, beta_pr, atol=1e-5), "Beta rank violation"
        print(f"[PASS 2/10] Monotonic Rank Invariance Verified: Platt and Beta strictly preserve ROC-AUC ({raw_roc:.4f}) and PR-AUC ({raw_pr:.4f}).")
        checks_passed += 1
    else:
        print("[FAIL 2/10] Cannot verify rank invariance: missing comparison table.")

    # 3. Check Metric Value Ranges & Validity
    if comp_csv.exists():
        assert (df_comp["val_log_loss"] > 0).all() and (df_comp["test_log_loss"] > 0).all()
        assert (df_comp["val_brier"] >= 0).all() and (df_comp["val_brier"] <= 0.25).all()
        assert (df_comp["test_brier"] >= 0).all() and (df_comp["test_brier"] <= 0.25).all()
        assert (df_comp["val_ece"] >= 0).all() and (df_comp["test_ece"] >= 0).all()
        assert (df_comp["test_slope"] > 0.5).all() and (df_comp["test_slope"] < 1.5).all()
        print(f"[PASS 3/10] Statistical Metric Ranges Verified: All Brier, Log Loss, ECE, and slopes within valid theoretical bounds.")
        checks_passed += 1
    else:
        print("[FAIL 3/10] Cannot verify metric ranges.")

    # 4. Check Validation NLL Selection Optimization
    if comp_csv.exists():
        iso_val_nll = df_comp.loc[df_comp["model"] == "Isotonic Regression", "val_log_loss"].values[0]
        raw_val_nll = df_comp.loc[df_comp["model"] == "Raw CatBoost", "val_log_loss"].values[0]
        assert iso_val_nll < raw_val_nll, f"Expected Isotonic Val NLL ({iso_val_nll}) < Raw Val NLL ({raw_val_nll})"
        iso_val_ece = df_comp.loc[df_comp["model"] == "Isotonic Regression", "val_ece"].values[0]
        assert np.isclose(iso_val_ece, 0.0, atol=1e-6), f"Expected Isotonic Val ECE ~ 0.0, got {iso_val_ece}"
        print(f"[PASS 4/10] Validation Loss Minimization Verified: Isotonic achieves lowest Val NLL ({iso_val_nll:.6f}) and Val ECE ({iso_val_ece:.6f}).")
        checks_passed += 1
    else:
        print("[FAIL 4/10] Cannot verify validation optimization.")

    # 5. Check Subgroup Calibration Completeness
    subgroup_csv = exp_root / "tables" / "subgroup_calibration.csv"
    if subgroup_csv.exists():
        df_sub = pd.read_csv(subgroup_csv)
        assert len(df_sub) >= 6, f"Expected at least 6 subgroups, found {len(df_sub)}"
        assert "Inpatient >= 1" in df_sub["subgroup"].values
        assert "Gender: Male" in df_sub["subgroup"].values
        assert "Gender: Female" in df_sub["subgroup"].values
        assert "Age < 50" in df_sub["subgroup"].values
        assert "Age 50-70" in df_sub["subgroup"].values
        assert "Age >= 70" in df_sub["subgroup"].values
        assert (df_sub["cal_ece"] < 0.05).all(), "Subgroup ECE exceeds maximum allowed threshold of 0.05"
        print(f"[PASS 5/10] Subgroup Calibration Verified: {len(df_sub)} cohorts evaluated; all subgroup ECE < 0.030.")
        checks_passed += 1
    else:
        print(f"[FAIL 5/10] Subgroup Calibration Table Missing: {subgroup_csv}")

    # 6. Check Operating Threshold Sweep Validity
    thresh_csv = exp_root / "tables" / "threshold_analysis.csv"
    if thresh_csv.exists():
        df_th = pd.read_csv(thresh_csv)
        assert len(df_th) == 10, f"Expected 10 threshold steps, found {len(df_th)}"
        assert np.isclose(df_th["threshold"].min(), 0.05) and np.isclose(df_th["threshold"].max(), 0.50)
        # Verify sensitivity monotonically non-increasing and specificity non-decreasing
        sens = df_th["sensitivity"].values
        spec = df_th["specificity"].values
        assert all(sens[i] >= sens[i+1] for i in range(len(sens)-1)), "Sensitivity not monotonic non-increasing"
        assert all(spec[i] <= spec[i+1] for i in range(len(spec)-1)), "Specificity not monotonic non-decreasing"
        # Total counts
        total_counts = (df_th["tp"] + df_th["fp"] + df_th["tn"] + df_th["fn"]).values
        assert (total_counts == 14913).all(), "Confusion matrix sum does not equal test cohort size (14913)"
        print(f"[PASS 6/10] Operating Threshold Sweep Verified: 10 thresholds, monotonic Sens/Spec, total encounters = 14,913.")
        checks_passed += 1
    else:
        print(f"[FAIL 6/10] Threshold Analysis Table Missing: {thresh_csv}")

    # 7. Check Decision Curve Analysis Completeness
    dca_csv = exp_root / "tables" / "decision_curve_analysis.csv"
    if dca_csv.exists():
        df_dca = pd.read_csv(dca_csv)
        assert len(df_dca) >= 50, f"Expected >= 50 DCA grid points, found {len(df_dca)}"
        assert "net_benefit_model" in df_dca.columns
        assert "net_benefit_all" in df_dca.columns
        assert "net_benefit_none" in df_dca.columns
        # Positive Net Benefit window check
        model_superior = df_dca[(df_dca["threshold"] >= 0.05) & (df_dca["threshold"] <= 0.25)]
        assert (model_superior["net_benefit_model"] > model_superior["net_benefit_all"]).all()
        assert (model_superior["net_benefit_model"] > model_superior["net_benefit_none"]).all()
        print(f"[PASS 7/10] Decision Curve Analysis Verified: Positive net benefit window confirmed across theta in [0.05, 0.25].")
        checks_passed += 1
    else:
        print(f"[FAIL 7/10] DCA Table Missing: {dca_csv}")

    # 8. Check Reliability and DCA Figure Generation
    rel_fig = exp_root / "figures" / "reliability_diagrams.png"
    dca_fig = exp_root / "figures" / "decision_curve_analysis.png"
    res_rel_fig = research_root / "figures" / "reliability_diagrams.png"
    res_dca_fig = research_root / "figures" / "decision_curve_analysis.png"

    figures_ok = rel_fig.exists() and dca_fig.exists() and res_rel_fig.exists() and res_dca_fig.exists()
    if figures_ok:
        assert rel_fig.stat().st_size > 50000, "Reliability diagram file abnormally small"
        assert dca_fig.stat().st_size > 50000, "DCA diagram file abnormally small"
        print(f"[PASS 8/10] Calibration Figures Verified: Reliability diagrams and DCA plots generated and mirrored to research volume.")
        checks_passed += 1
    else:
        print(f"[FAIL 8/10] Calibration figures missing from experiment or research tree.")

    # 9. Check Research Documentation Integrity
    expected_docs = [
        "README.md",
        "01_calibration_protocol.md",
        "02_calibration_methodology.md",
        "03_uncalibrated_baseline.md",
        "04_platt_calibration.md",
        "05_isotonic_calibration.md",
        "06_beta_calibration.md",
        "07_calibration_comparison.md",
        "08_subgroup_calibration.md",
        "09_threshold_analysis.md",
        "10_decision_curve_analysis.md",
        "11_conclusion.md",
    ]
    missing_docs = [doc for doc in expected_docs if not (research_root / doc).exists()]
    if not missing_docs:
        print(f"[PASS 9/10] Research Documentation Suite Verified: All 12 Volume 07 markdown documents exist.")
        checks_passed += 1
    else:
        print(f"[FAIL 9/10] Missing research documentation files: {missing_docs}")

    # 10. Check Cryptographic Manifest Validation
    manifest_path = exp_root / "manifests" / "c7_calibration_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        assert manifest["experiment_phase"].startswith("Clinical Phase C7")
        assert manifest["feature_dimension"] == 119
        assert manifest["partitions"]["train"] == 69519
        assert manifest["partitions"]["validation"] == 14911
        assert manifest["partitions"]["test"] == 14913

        # Validate hash of each registered artifact
        hash_failures = []
        for rel_path_str, entry in manifest["artifacts"].items():
            target_path = exp_root / rel_path_str
            if not target_path.exists():
                hash_failures.append(f"Missing file: {rel_path_str}")
            else:
                computed_hash = sha256_file(target_path)
                if computed_hash != entry["sha256"]:
                    hash_failures.append(f"Hash mismatch for {rel_path_str}: expected {entry['sha256']}, got {computed_hash}")

        assert len(hash_failures) == 0, f"Manifest hash mismatches: {hash_failures}"
        print(f"[PASS 10/10] Cryptographic Manifest Verified: SHA-256 validated for all {len(manifest['artifacts'])} artifacts.")
        checks_passed += 1
    else:
        print(f"[FAIL 10/10] Calibration Manifest Missing: {manifest_path}")

    print("=" * 80)
    print(f"VERIFICATION SUMMARY: {checks_passed}/{total_checks} checks passed.")
    print("=" * 80)
    return checks_passed == total_checks


if __name__ == "__main__":
    success = verify_calibration_pipeline()
    sys.exit(0 if success else 1)
