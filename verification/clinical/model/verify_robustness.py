"""Phase C9 Clinical Robustness, Fairness & Distribution Shift Verification Suite."""

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


def verify_robustness_pipeline() -> bool:
    print("=" * 80)
    print("VERIFICATION: Clinical Phase C9 Robustness & Distribution Shift Auditing")
    print("=" * 80)

    exp_root = get_runtime_output_root(REPO_ROOT, "robustness")
    research_root = REPO_ROOT / "research" / "clinical" / "Volume_09_Robustness_Fairness"

    checks_passed = 0
    total_checks = 10

    # 1. Check Frozen Model and Preprocessor Contract
    manifest_path = exp_root / "manifests" / "c9_robustness_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        assert manifest["frozen_model"] == "CatBoost_HPO_Tuned_Candidate"
        assert manifest["frozen_calibration"] == "Isotonic_Regression"
        assert manifest["ensemble_size_M"] == 50
        assert manifest["partitions"]["train"] == 69519
        assert manifest["partitions"]["validation"] == 14911
        assert manifest["partitions"]["test"] == 14913
        print("[PASS  1/10] Frozen Model & Pipeline Invariants Verified (CatBoost HPO + Isotonic Calibrator).")
        checks_passed += 1
    else:
        print(f"[FAIL  1/10] Manifest Missing: {manifest_path}")

    # 2. Check Nominal Reference Metrics
    nom_csv = exp_root / "tables" / "nominal_reference.csv"
    if nom_csv.exists():
        df_nom = pd.read_csv(nom_csv)
        nom_dict = dict(zip(df_nom["metric"], df_nom["value"]))
        assert float(nom_dict["roc_auc"]) >= 0.640, f"Nominal ROC-AUC below 0.640: {nom_dict['roc_auc']}"
        assert float(nom_dict["pr_auc"]) >= 0.190, f"Nominal PR-AUC below 0.190: {nom_dict['pr_auc']}"
        assert float(nom_dict["mean_uncertainty"]) > 0.015
        print(f"[PASS  2/10] Nominal Reference Profile Verified (ROC-AUC={nom_dict['roc_auc']}, PR-AUC={nom_dict['pr_auc']}).")
        checks_passed += 1
    else:
        print(f"[FAIL  2/10] Nominal Reference CSV Missing: {nom_csv}")

    # 3. Check Missingness-Shift Experiments
    miss_csv = exp_root / "tables" / "missingness_degradation.csv"
    if miss_csv.exists():
        df_miss = pd.read_csv(miss_csv)
        assert len(df_miss) >= 7, f"Expected >= 7 missingness scenarios, found {len(df_miss)}"
        assert "M0_Nominal_Reference" in df_miss["scenario"].values
        assert "M1_Random_MCAR_10pct" in df_miss["scenario"].values
        assert "M2_Random_MCAR_25pct" in df_miss["scenario"].values
        assert "M3_Random_MCAR_50pct" in df_miss["scenario"].values
        
        # Verify uncertainty inflation under MCAR
        u_m0 = df_miss.loc[df_miss["scenario"] == "M0_Nominal_Reference", "mean_uncertainty"].values[0]
        u_m3 = df_miss.loc[df_miss["scenario"] == "M3_Random_MCAR_50pct", "mean_uncertainty"].values[0]
        assert u_m3 > u_m0 * 1.5, f"Expected 50% MCAR uncertainty ({u_m3}) > 1.5x nominal ({u_m0})"
        print(f"[PASS  3/10] Missingness-Shift Suite Verified: MCAR inflation confirmed ({u_m0:.4f} -> {u_m3:.4f}).")
        checks_passed += 1
    else:
        print(f"[FAIL  3/10] Missingness Table Missing: {miss_csv}")

    # 4. Check Demographic & Subgroup Shift
    demo_csv = exp_root / "tables" / "demographic_subgroup_shift.csv"
    pop_csv = exp_root / "tables" / "population_composition_shift.csv"
    if demo_csv.exists() and pop_csv.exists():
        df_demo = pd.read_csv(demo_csv)
        df_pop = pd.read_csv(pop_csv)
        assert len(df_demo) >= 6, f"Expected >= 6 demographic subgroups, got {len(df_demo)}"
        assert len(df_pop) >= 4, f"Expected >= 4 composition shifts, got {len(df_pop)}"
        assert "Gender_Female" in df_demo["subgroup_id"].values
        assert "Age_Younger_lt50" in df_demo["subgroup_id"].values
        print(f"[PASS  4/10] Demographic & Population Shifts Verified ({len(df_demo)} strata, {len(df_pop)} composition shifts).")
        checks_passed += 1
    else:
        print("[FAIL  4/10] Demographic or Population CSV Missing.")

    # 5. Check Encounter Complexity & Utilization Shifts
    strata_csv = exp_root / "tables" / "encounter_strata_shift.csv"
    enc_shift_csv = exp_root / "tables" / "encounter_distribution_shift.csv"
    if strata_csv.exists() and enc_shift_csv.exists():
        df_strata = pd.read_csv(strata_csv)
        df_enc = pd.read_csv(enc_shift_csv)
        assert len(df_strata) >= 6, f"Expected >= 6 complexity strata, got {len(df_strata)}"
        assert "Util_Zero_Inpatient" in df_strata["stratum_id"].values
        assert "Util_Frequent_Inpatient" in df_strata["stratum_id"].values
        
        # Verify utilization uncertainty divergence
        u_zero = df_strata.loc[df_strata["stratum_id"] == "Util_Zero_Inpatient", "mean_uncertainty"].values[0]
        u_freq = df_strata.loc[df_strata["stratum_id"] == "Util_Frequent_Inpatient", "mean_uncertainty"].values[0]
        assert u_freq > u_zero * 2.0, f"Expected Frequent Inpatient uncertainty ({u_freq}) > 2x Zero Inpatient ({u_zero})"
        print(f"[PASS  5/10] Encounter Complexity Shifts Verified (Frequent vs Zero Inpatient: {u_freq:.4f} vs {u_zero:.4f}).")
        checks_passed += 1
    else:
        print("[FAIL  5/10] Encounter Strata CSV Missing.")

    # 6. Check Temporal Robustness & Chronological Drift
    temp_csv = exp_root / "tables" / "temporal_shift.csv"
    if temp_csv.exists():
        df_temp = pd.read_csv(temp_csv)
        assert len(df_temp) >= 6, f"Expected >= 6 temporal slices, got {len(df_temp)}"
        assert "Temporal_Early_Era" in df_temp["temporal_id"].values
        assert "Temporal_Late_Era" in df_temp["temporal_id"].values
        assert "Temporal_Q1_Earliest" in df_temp["temporal_id"].values
        print(f"[PASS  6/10] Temporal Sequence Stability Verified (Early vs Late Era documented).")
        checks_passed += 1
    else:
        print(f"[FAIL  6/10] Temporal Shift Table Missing: {temp_csv}")

    # 7. Check Epistemic Failure Regimes & High-Confidence Silent Errors
    fail_csv = exp_root / "tables" / "failure_regimes_high_confidence_errors.csv"
    if fail_csv.exists():
        df_fail = pd.read_csv(fail_csv)
        assert len(df_fail) == 4, f"Expected 4 epistemic regimes, got {len(df_fail)}"
        assert "Q1: Confident Correct" in df_fail["regime_name"].values
        assert "Q4: High-Confidence Silent Failure" in df_fail["regime_name"].values
        
        q4_n = df_fail.loc[df_fail["regime_name"] == "Q4: High-Confidence Silent Failure", "encounter_count"].values[0]
        assert q4_n > 0, "No Q4 silent errors recorded"
        print(f"[PASS  7/10] Epistemic Failure Regimes Verified (Q4 Silent Failures = {q4_n:,} encounters).")
        checks_passed += 1
    else:
        print(f"[FAIL  7/10] Failure Regimes CSV Missing: {fail_csv}")

    # 8. Check Master Robustness Summary Matrix
    matrix_csv = exp_root / "tables" / "robustness_summary_matrix.csv"
    if matrix_csv.exists():
        df_mat = pd.read_csv(matrix_csv)
        assert len(df_mat) >= 15, f"Expected >= 15 matrix rows, got {len(df_mat)}"
        assert "Nominal Reference" in df_mat["category"].values
        assert "Missingness Perturbation" in df_mat["category"].values
        assert "Demographic Cohort" in df_mat["category"].values
        print(f"[PASS  8/10] Master Robustness Summary Matrix Verified ({len(df_mat)} evaluated scenarios).")
        checks_passed += 1
    else:
        print(f"[FAIL  8/10] Robustness Summary Matrix Missing: {matrix_csv}")

    # 9. Check Publication Figures & Research Volume Sync
    fig_names = [
        "missingness_degradation.png",
        "subgroup_shift.png",
        "encounter_shift.png",
        "temporal_shift.png",
        "uncertainty_shift.png",
        "risk_coverage_shift.png",
        "robustness_summary.png",
    ]
    figs_exp_ok = all((exp_root / "figures" / f).exists() for f in fig_names)
    figs_res_ok = all((research_root / "figures" / f).exists() for f in fig_names)
    
    expected_docs = [
        "README.md",
        "01_robustness_protocol.md",
        "02_distribution_shift_methodology.md",
        "03_missingness_shift.md",
        "04_demographic_shift.md",
        "05_encounter_distribution_shift.md",
        "06_temporal_shift.md",
        "07_uncertainty_under_shift.md",
        "08_risk_coverage_under_shift.md",
        "09_subgroup_robustness.md",
        "10_failure_regime_analysis.md",
        "11_robustness_comparison.md",
        "12_conclusion.md",
    ]
    docs_ok = all((research_root / d).exists() for d in expected_docs)

    if figs_exp_ok and figs_res_ok:
        print(f"[PASS  9/10] Publication Figures Verified: 7 publication figures present in experiments and mirrored.")
        checks_passed += 1
    else:
        print(f"[FAIL  9/10] Missing figures in experiments or research volume.")

    # 10. Check Cryptographic Manifest
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        assert manifest["experiment_phase"].startswith("Clinical Phase C9")

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
    success = verify_robustness_pipeline()
    sys.exit(0 if success else 1)
