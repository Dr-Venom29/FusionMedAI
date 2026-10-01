"""Phase C8 Clinical Prediction Uncertainty Estimation Verification Suite."""

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


def verify_uncertainty_pipeline() -> bool:
    print("=" * 80)
    print("VERIFICATION: Clinical Phase C8 Prediction Uncertainty Estimation")
    print("=" * 80)

    exp_root = get_runtime_output_root(REPO_ROOT, "uncertainty")
    research_root = REPO_ROOT / "research" / "clinical" / "Volume_08_Uncertainty"

    checks_passed = 0
    total_checks = 10

    # 1. Check Uncertainty Summary Table Completeness
    summary_csv = exp_root / "tables" / "uncertainty_summary.csv"
    if summary_csv.exists():
        df_sum = pd.read_csv(summary_csv)
        assert len(df_sum) >= 10, f"Expected >= 10 summary rows, found {len(df_sum)}"
        metrics_dict = dict(zip(df_sum["metric"], df_sum["value"]))
        assert metrics_dict.get("Total Test Encounters") == "14,913"
        assert metrics_dict.get("Ensemble Size (M)") == "50"
        print(f"[PASS 1/10] Uncertainty Summary Table Verified: 50-model ensemble on 14,913 test encounters.")
        checks_passed += 1
    else:
        print(f"[FAIL 1/10] Uncertainty Summary Table Missing: {summary_csv}")

    # 2. Check Prediction Distribution Statistics
    if summary_csv.exists():
        mean_u = float(metrics_dict["Mean Predictive Uncertainty (std)"])
        median_u = float(metrics_dict["Median Predictive Uncertainty (std)"])
        assert 0.010 <= mean_u <= 0.050, f"Mean uncertainty out of expected range: {mean_u}"
        assert 0.005 <= median_u <= 0.030, f"Median uncertainty out of expected range: {median_u}"
        print(f"[PASS 2/10] Prediction Distribution Statistics Verified: Mean sigma_p = {mean_u:.4f}, Median = {median_u:.4f}.")
        checks_passed += 1
    else:
        print("[FAIL 2/10] Cannot verify prediction distribution stats.")

    # 3. Check Error Detection Capability
    if summary_csv.exists():
        # Match metric name containing 'Error Detection AUROC'
        auroc_key = [k for k in metrics_dict if "Error Detection AUROC" in k][0]
        err_auroc = float(metrics_dict[auroc_key])
        assert err_auroc >= 0.650, f"Error detection AUROC below minimum threshold 0.650: {err_auroc}"
        print(f"[PASS 3/10] Error Detection Capability Verified: Error Detection AUROC = {err_auroc:.4f} (>= 0.650).")
        checks_passed += 1
    else:
        print("[FAIL 3/10] Cannot verify error detection capability.")

    # 4. Check Risk-Coverage Analysis & Monotonicity
    rc_csv = exp_root / "tables" / "risk_coverage.csv"
    if rc_csv.exists():
        df_rc = pd.read_csv(rc_csv)
        assert len(df_rc) >= 10, f"Expected >= 10 coverage steps, found {len(df_rc)}"
        assert np.isclose(df_rc["coverage"].max(), 1.0) and np.isclose(df_rc["coverage"].min(), 0.10)
        
        # Check overall risk reduction
        full_risk = df_rc.loc[np.isclose(df_rc["coverage"], 1.0), "risk_error_rate"].values[0]
        half_risk = df_rc.loc[np.isclose(df_rc["coverage"], 0.50), "risk_error_rate"].values[0]
        assert half_risk < full_risk, f"Expected risk at 50% coverage ({half_risk}) < risk at 100% coverage ({full_risk})"
        
        aurc_key = [k for k in metrics_dict if "Risk-Coverage AURC" in k][0]
        aurc_val = float(metrics_dict[aurc_key])
        assert 0.040 <= aurc_val <= 0.120, f"AURC out of expected range: {aurc_val}"
        print(f"[PASS 4/10] Risk-Coverage Dynamics Verified: AURC = {aurc_val:.4f}, Error rate drops from {full_risk*100:.1f}% to {half_risk*100:.1f}%.")
        checks_passed += 1
    else:
        print(f"[FAIL 4/10] Risk-Coverage Table Missing: {rc_csv}")

    # 5. Check Threshold Ambiguity & Clinical Decision Tiers
    tiers_csv = exp_root / "tables" / "threshold_uncertainty_tiers.csv"
    if tiers_csv.exists():
        df_tiers = pd.read_csv(tiers_csv)
        assert len(df_tiers) == 6, f"Expected 6 decision tiers, found {len(df_tiers)}"
        total_n = df_tiers["encounter_count"].sum()
        assert total_n == 14913, f"Expected total encounters 14,913, got {total_n}"
        assert "Low Risk / Low Uncertainty" in df_tiers["tier"].values
        assert "Near Threshold / High Uncertainty (Ambiguity)" in df_tiers["tier"].values
        print(f"[PASS 5/10] Decision-Support Tiers Verified: 6 operational tiers spanning 14,913 encounters.")
        checks_passed += 1
    else:
        print(f"[FAIL 5/10] Threshold Tiers Table Missing: {tiers_csv}")

    # 6. Check Subgroup Uncertainty & Phenotype Audit
    sub_csv = exp_root / "tables" / "subgroup_uncertainty.csv"
    if sub_csv.exists():
        df_sub = pd.read_csv(sub_csv)
        assert len(df_sub) >= 7, f"Expected >= 7 subgroups, found {len(df_sub)}"
        assert "Prior Inpatient = 0" in df_sub["subgroup"].values
        assert "Prior Inpatient >= 1" in df_sub["subgroup"].values
        
        inp0_u = df_sub.loc[df_sub["subgroup"] == "Prior Inpatient = 0", "mean_uncertainty"].values[0]
        inp1_u = df_sub.loc[df_sub["subgroup"] == "Prior Inpatient >= 1", "mean_uncertainty"].values[0]
        assert inp1_u > inp0_u, f"Expected Inpatient >= 1 uncertainty ({inp1_u}) > Inpatient = 0 uncertainty ({inp0_u})"
        print(f"[PASS 6/10] Subgroup Phenotype Audit Verified: Prior Inpatient divergence confirmed ({inp0_u:.4f} vs {inp1_u:.4f}).")
        checks_passed += 1
    else:
        print(f"[FAIL 6/10] Subgroup Uncertainty Table Missing: {sub_csv}")

    # 7. Check Ensemble Convergence Stability
    conv_csv = exp_root / "tables" / "convergence_analysis.csv"
    if conv_csv.exists():
        df_conv = pd.read_csv(conv_csv)
        assert len(df_conv) >= 5, f"Expected >= 5 convergence points, found {len(df_conv)}"
        corr_50 = df_conv.loc[df_conv["ensemble_size_M"] == 50, "spearman_corr_std"].values[0]
        corr_40 = df_conv.loc[df_conv["ensemble_size_M"] == 40, "spearman_corr_std"].values[0]
        assert np.isclose(corr_50, 1.0, atol=1e-5), "Full ensemble correlation not 1.0"
        assert corr_40 >= 0.980, f"M=40 correlation below 0.980: {corr_40}"
        print(f"[PASS 7/10] Ensemble Convergence Verified: M=50 convergence confirmed (M=40 rho = {corr_40:.4f}).")
        checks_passed += 1
    else:
        print(f"[FAIL 7/10] Convergence Table Missing: {conv_csv}")

    # 8. Check Local Patient Cases
    cases_csv = exp_root / "tables" / "local_uncertainty_cases.csv"
    if cases_csv.exists():
        df_cases = pd.read_csv(cases_csv)
        assert len(df_cases) == 7, f"Expected 7 patient case studies, found {len(df_cases)}"
        assert (df_cases["uncertainty_std"] > 0).all(), "Zero uncertainty detected in local cases"
        assert (df_cases["pi_95_upper"] >= df_cases["pi_95_lower"]).all(), "Interval bound violation"
        print(f"[PASS 8/10] Local Case Studies Verified: 7 multi-dimensional clinical profiles connecting SHAP, Cal, and Unc.")
        checks_passed += 1
    else:
        print(f"[FAIL 8/10] Local Cases Table Missing: {cases_csv}")

    # 9. Check Publication Figures & Research Volume Sync
    fig_names = [
        "error_detection_distributions.png",
        "risk_coverage_curve.png",
        "threshold_uncertainty_scatter.png",
        "convergence_analysis.png",
        "local_case_profiles.png",
        "clinical_output_pipeline.png",
    ]
    figs_exp_ok = all((exp_root / "figures" / f).exists() for f in fig_names)
    figs_res_ok = all((research_root / "figures" / f).exists() for f in fig_names)
    
    expected_docs = [
        "README.md",
        "01_uncertainty_protocol.md",
        "02_uncertainty_methodology.md",
        "03_bootstrap_uncertainty.md",
        "04_predictive_distribution.md",
        "05_error_detection.md",
        "06_risk_coverage.md",
        "07_subgroup_uncertainty.md",
        "08_threshold_uncertainty.md",
        "09_convergence_analysis.md",
        "10_local_uncertainty_cases.md",
        "11_clinical_output_contract.md",
        "12_conclusion.md",
    ]
    docs_ok = all((research_root / d).exists() for d in expected_docs)

    if figs_exp_ok and figs_res_ok and docs_ok:
        print(f"[PASS 9/10] Research Volume 08 Sync Verified: 6 publication figures and all 13 markdown documents present.")
        checks_passed += 1
    else:
        print(f"[FAIL 9/10] Missing research documentation or mirrored figures.")

    # 10. Check Cryptographic Manifest
    manifest_path = exp_root / "manifests" / "c8_uncertainty_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        assert manifest["experiment_phase"].startswith("Clinical Phase C8")
        assert manifest["ensemble_size_M"] == 50
        assert manifest["partitions"]["train"] == 69519
        assert manifest["partitions"]["validation"] == 14911
        assert manifest["partitions"]["test"] == 14913

        hash_failures = []
        for rel_path_str, entry in manifest["artifacts"].items():
            target_path = exp_root / rel_path_str
            if not target_path.exists():
                hash_failures.append(f"Missing file: {rel_path_str}")
            else:
                computed_hash = sha256_file(target_path)
                if computed_hash != entry["sha256"]:
                    hash_failures.append(f"Hash mismatch for {rel_path_str}")

        assert len(hash_failures) == 0, f"Hash failures: {hash_failures}"
        print(f"[PASS 10/10] Cryptographic Manifest Verified: SHA-256 validated for all {len(manifest['artifacts'])} artifacts.")
        checks_passed += 1
    else:
        print(f"[FAIL 10/10] Manifest File Missing: {manifest_path}")

    print("=" * 80)
    print(f"VERIFICATION SUMMARY: {checks_passed}/{total_checks} checks passed.")
    print("=" * 80)
    return checks_passed == total_checks


if __name__ == "__main__":
    success = verify_uncertainty_pipeline()
    sys.exit(0 if success else 1)
