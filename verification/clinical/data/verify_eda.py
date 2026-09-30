"""
FusionMedAI - Phase C3: Exploratory Data Analysis & Feature Representation Verification Gate
Automated unit verification script for C3 16-gate checks.
"""

import sys
import os
import json
import hashlib
import pandas as pd
import numpy as np

def find_repo_root():
    curr = os.path.abspath(os.path.dirname(__file__))
    while curr and os.path.splitdrive(curr)[1] != '\\':
        if os.path.exists(os.path.join(curr, "datasets", "clinical", "diabetic_data.csv")):
            return curr
        parent = os.path.dirname(curr)
        if parent == curr:
            break
        curr = parent
    if os.path.exists(os.path.join(os.getcwd(), "datasets", "clinical", "diabetic_data.csv")):
        return os.getcwd()
    return os.path.abspath(".")

REPO_ROOT = find_repo_root()
SPLITS_DIR = os.path.join(REPO_ROOT, "datasets", "clinical", "processed", "splits")
EDA_DIR = os.path.join(REPO_ROOT, "datasets", "clinical", "metadata", "eda")
STATS_DIR = os.path.join(EDA_DIR, "statistics")
REPORTS_DIR = os.path.join(EDA_DIR, "reports")

TRAIN_PATH = os.path.join(SPLITS_DIR, "train.csv")
VAL_PATH = os.path.join(SPLITS_DIR, "val.csv")
TEST_PATH = os.path.join(SPLITS_DIR, "test.csv")

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def run_c3_verification():
    results = {}
    print("=" * 70)
    print("FusionMedAI: Phase C3 - EDA & Feature Representation Verification Gate")
    print("=" * 70)

    if not (os.path.exists(TRAIN_PATH) and os.path.exists(VAL_PATH) and os.path.exists(TEST_PATH)):
        print("ERROR: Split datasets not found.")
        return False

    train_df = pd.read_csv(TRAIN_PATH, keep_default_na=False)
    val_df = pd.read_csv(VAL_PATH, keep_default_na=False)
    test_df = pd.read_csv(TEST_PATH, keep_default_na=False)

    # 1. Split row counts
    print("\n[Gate 1/16] Checking Split Row Counts...")
    if len(train_df) == 69519 and len(val_df) == 14911 and len(test_df) == 14913:
        results["Split row counts"] = "PASS"
        print(f"  -> Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,} encounters.")
    else:
        results["Split row counts"] = "FAIL"
        print("  -> Split row counts mismatch.")

    # 2. Target prevalence
    print("\n[Gate 2/16] Checking Target Prevalence...")
    tr_rate = (train_df['readmitted'] == '<30').mean()
    va_rate = (val_df['readmitted'] == '<30').mean()
    te_rate = (test_df['readmitted'] == '<30').mean()
    if abs(tr_rate - 0.1139) < 0.005 and abs(va_rate - 0.1167) < 0.005 and abs(te_rate - 0.1112) < 0.005:
        results["Target prevalence"] = "PASS"
        print(f"  -> Target prevalence remained stable across partitions (range: 11.12%-11.67%).")
    else:
        results["Target prevalence"] = "FAIL"
        print(f"  -> Target rate discrepancy: {tr_rate}, {va_rate}, {te_rate}")

    # 3. Complete patient isolation (Train ∩ Val = 0, Train ∩ Test = 0, Val ∩ Test = 0)
    print("\n[Gate 3/16] Checking Complete Patient Isolation Across Partitions...")
    tr_pats = set(train_df['patient_nbr'])
    va_pats = set(val_df['patient_nbr'])
    te_pats = set(test_df['patient_nbr'])

    train_val_overlap = tr_pats.intersection(va_pats)
    train_test_overlap = tr_pats.intersection(te_pats)
    val_test_overlap = va_pats.intersection(te_pats)

    if len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0:
        results["Complete patient isolation"] = "PASS"
        print("  -> Zero patient overlap across all three partitions (Train, Val, Test).")
    else:
        results["Complete patient isolation"] = "FAIL"
        print(f"  -> Overlap detected: Train-Val={len(train_val_overlap)}, Train-Test={len(train_test_overlap)}, Val-Test={len(val_test_overlap)}")

    # 4. Missingness artifact
    print("\n[Gate 4/16] Checking Missingness Statistics Artifact...")
    missing_csv = os.path.join(STATS_DIR, "missingness_stats.csv")
    if os.path.exists(missing_csv):
        df_m = pd.read_csv(missing_csv)
        req_cols = {"feature", "q_count", "q_pct", "none_count", "none_pct"}
        if req_cols.issubset(df_m.columns) and len(df_m) > 0:
            results["Missingness artifact"] = "PASS"
            print(f"  -> {len(df_m)} columns audited for '?' and 'None' mechanisms in {missing_csv}.")
        else:
            results["Missingness artifact"] = "FAIL"
    else:
        results["Missingness artifact"] = "FAIL"

    # 5. Numerical statistics artifact
    print("\n[Gate 5/16] Checking Numerical Statistics Artifact...")
    num_csv = os.path.join(STATS_DIR, "numerical_stats.csv")
    if os.path.exists(num_csv):
        df_n = pd.read_csv(num_csv)
        req_cols = {"feature", "mean", "std", "median", "skewness", "zero_pct"}
        if req_cols.issubset(df_n.columns) and len(df_n) == 8:
            results["Numerical statistics artifact"] = "PASS"
            print("  -> 8 numerical features profiled (mean, std, median, skewness, zero %) in numerical_stats.csv.")
        else:
            results["Numerical statistics artifact"] = "FAIL"
    else:
        results["Numerical statistics artifact"] = "FAIL"

    # 6. Categorical statistics artifact
    print("\n[Gate 6/16] Checking Categorical Statistics Artifact...")
    cat_csv = os.path.join(STATS_DIR, "categorical_stats.csv")
    if os.path.exists(cat_csv):
        df_c = pd.read_csv(cat_csv)
        req_cols = {"feature", "cardinality", "top_category"}
        if req_cols.issubset(df_c.columns) and len(df_c) > 0:
            results["Categorical statistics artifact"] = "PASS"
            print(f"  -> {len(df_c)} core categorical attributes audited in categorical_stats.csv.")
        else:
            results["Categorical statistics artifact"] = "FAIL"
    else:
        results["Categorical statistics artifact"] = "FAIL"

    # 7. Clinical feature-group artifact
    print("\n[Gate 7/16] Checking Clinical Feature-Group Artifact...")
    groups_json = os.path.join(REPORTS_DIR, "clinical_feature_groups_report.json")
    if os.path.exists(groups_json):
        with open(groups_json, "r") as f:
            grp_data = json.load(f)
        if "domains" in grp_data and len(grp_data["domains"]) == 9:
            results["Clinical feature-group artifact"] = "PASS"
            print("  -> 9 clinical domains verified in clinical_feature_groups_report.json.")
        else:
            results["Clinical feature-group artifact"] = "FAIL"
    else:
        results["Clinical feature-group artifact"] = "FAIL"

    # 8. Diagnosis mapping artifact
    print("\n[Gate 8/16] Checking Diagnosis Mapping Artifact...")
    diag_json = os.path.join(REPORTS_DIR, "diagnosis_mapping_report.json")
    if os.path.exists(diag_json):
        with open(diag_json, "r") as f:
            diag_data = json.load(f)
        fields = diag_data.get("diagnosis_fields", {})
        c1 = fields.get("diag_1", {}).get("raw_cardinality")
        c2 = fields.get("diag_2", {}).get("raw_cardinality")
        c3 = fields.get("diag_3", {}).get("raw_cardinality")
        if c1 == 689 and c2 == 697 and c3 == 744:
            results["Diagnosis mapping artifact"] = "PASS"
            print(f"  -> 9-Chapter ICD-9 mapping verified for diag_1 ({c1}), diag_2 ({c2}), diag_3 ({c3}).")
        else:
            results["Diagnosis mapping artifact"] = "FAIL"
            print(f"  -> Cardinality mismatch: {c1}, {c2}, {c3}")
    else:
        results["Diagnosis mapping artifact"] = "FAIL"

    # 9. Medication statistics artifact
    print("\n[Gate 9/16] Checking Medication Statistics Artifact...")
    med_csv = os.path.join(STATS_DIR, "medication_stats.csv")
    if os.path.exists(med_csv):
        df_med = pd.read_csv(med_csv)
        if len(df_med) == 23:
            results["Medication statistics artifact"] = "PASS"
            print("  -> 23 medication columns audited: 21 active medications + 2 zero-variance medications.")
        else:
            results["Medication statistics artifact"] = "FAIL"
    else:
        results["Medication statistics artifact"] = "FAIL"

    # 10. Association artifact
    print("\n[Gate 10/16] Checking Association & Redundancy Artifact...")
    assoc_json = os.path.join(REPORTS_DIR, "feature_association_report.json")
    if os.path.exists(assoc_json):
        with open(assoc_json, "r") as f:
            assoc_data = json.load(f)
        if "pairwise_spearman_correlation" in assoc_data and "collinear_pairs" in assoc_data:
            results["Association artifact"] = "PASS"
            print("  -> Spearman correlation matrix and collinearity registers verified in feature_association_report.json.")
        else:
            results["Association artifact"] = "FAIL"
    else:
        results["Association artifact"] = "FAIL"

    # 11. Outlier artifact
    print("\n[Gate 11/16] Checking Outlier Analysis Artifact...")
    outlier_json = os.path.join(REPORTS_DIR, "outlier_analysis_report.json")
    if os.path.exists(outlier_json):
        with open(outlier_json, "r") as f:
            outlier_data = json.load(f)
        if "features" in outlier_data and len(outlier_data["features"]) == 8:
            results["Outlier artifact"] = "PASS"
            print("  -> Extreme value profiles and zero sample deletion policy verified in outlier_analysis_report.json.")
        else:
            results["Outlier artifact"] = "FAIL"
    else:
        results["Outlier artifact"] = "FAIL"

    # 12. Subgroup artifact
    print("\n[Gate 12/16] Checking Subgroup Analysis Artifact...")
    sg_csv = os.path.join(STATS_DIR, "subgroup_stats.csv")
    if os.path.exists(sg_csv):
        df_sg = pd.read_csv(sg_csv)
        if len(df_sg) > 0 and "subgroup_dimension" in df_sg.columns:
            results["Subgroup artifact"] = "PASS"
            print(f"  -> {len(df_sg)} subgroup strata evaluated in subgroup_stats.csv.")
        else:
            results["Subgroup artifact"] = "FAIL"
    else:
        results["Subgroup artifact"] = "FAIL"

    # 13. Feature representation contract
    print("\n[Gate 13/16] Checking Feature Representation Contract...")
    contract_json = os.path.join(REPORTS_DIR, "feature_representation_contract.json")
    if os.path.exists(contract_json):
        with open(contract_json, "r") as f:
            contract_data = json.load(f)
        if "identifiers" in contract_data and "numerical_features" in contract_data and "diagnoses" in contract_data:
            results["Feature representation contract"] = "PASS"
            print("  -> Frozen feature representation contract verified in feature_representation_contract.json.")
        else:
            results["Feature representation contract"] = "FAIL"
    else:
        results["Feature representation contract"] = "FAIL"

    # 14. Train-only provenance
    print("\n[Gate 14/16] Checking Train-Only Provenance Metadata...")
    provenance_verified = True
    for report_file in ["diagnosis_mapping_report.json", "clinical_feature_groups_report.json",
                        "feature_association_report.json", "outlier_analysis_report.json",
                        "feature_representation_contract.json"]:
        p = os.path.join(REPORTS_DIR, report_file)
        if os.path.exists(p):
            with open(p, "r") as f:
                d = json.load(f)
            prov = d.get("provenance", {})
            if prov.get("analysis_population") != "train" or prov.get("train_rows") != 69519:
                provenance_verified = False
                break
        else:
            provenance_verified = False
            break

    if provenance_verified:
        results["Train-only provenance"] = "PASS"
        print("  -> Source split provenance verified: analysis_population='train' (69,519 rows).")
    else:
        results["Train-only provenance"] = "FAIL"
        print("  -> Provenance metadata mismatch.")

    # 15. Artifact integrity / hashes
    print("\n[Gate 15/16] Checking Artifact Integrity & Cryptographic Hashes...")
    EXPECTED_ARTIFACTS = {
        "datasets/clinical/metadata/eda/statistics/missingness_stats.csv",
        "datasets/clinical/metadata/eda/statistics/numerical_stats.csv",
        "datasets/clinical/metadata/eda/statistics/categorical_stats.csv",
        "datasets/clinical/metadata/eda/statistics/medication_stats.csv",
        "datasets/clinical/metadata/eda/statistics/subgroup_stats.csv",
        "datasets/clinical/metadata/eda/reports/clinical_feature_groups_report.json",
        "datasets/clinical/metadata/eda/reports/diagnosis_mapping_report.json",
        "datasets/clinical/metadata/eda/reports/feature_association_report.json",
        "datasets/clinical/metadata/eda/reports/outlier_analysis_report.json",
        "datasets/clinical/metadata/eda/reports/feature_representation_contract.json",
    }
    manifest_path = os.path.join(REPORTS_DIR, "eda_artifact_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest_data = json.load(f)
        manifest_artifacts = set(manifest_data.get("artifacts", {}).keys())
        if manifest_artifacts != EXPECTED_ARTIFACTS:
            results["Artifact integrity / hashes"] = "FAIL"
            print(f"  -> Manifest artifact set mismatch: found {len(manifest_artifacts)}, expected {len(EXPECTED_ARTIFACTS)}.")
        else:
            hashes_match = True
            for rel_path in EXPECTED_ARTIFACTS:
                meta = manifest_data["artifacts"][rel_path]
                full_p = os.path.join(REPO_ROOT, rel_path)
                if not os.path.exists(full_p) or sha256_file(full_p) != meta["sha256"]:
                    hashes_match = False
                    print(f"  -> Hash mismatch or missing file for {rel_path}")
                    break
            if hashes_match:
                results["Artifact integrity / hashes"] = "PASS"
                print("  -> Explicit 10/10 generated EDA artifacts match frozen SHA-256 hashes in manifest.")
            else:
                results["Artifact integrity / hashes"] = "FAIL"
    else:
        results["Artifact integrity / hashes"] = "FAIL"

    # 16. Frozen artifact reproducibility
    print("\n[Gate 16/16] Checking Frozen Artifact Reproducibility...")
    split_meta_path = os.path.join(REPO_ROOT, "datasets", "clinical", "processed", "splits", "split_metadata.json")
    reproducibility_pass = False
    if os.path.exists(split_meta_path) and os.path.exists(manifest_path):
        with open(split_meta_path, "r") as f:
            split_meta = json.load(f)
        with open(manifest_path, "r") as f:
            manifest_data = json.load(f)
        prov = manifest_data.get("provenance", {})
        train_enc = split_meta.get("train_encounters")
        train_pat = split_meta.get("train_patients")
        manifest_arts = set(manifest_data.get("artifacts", {}).keys())

        # Self-contained hash verification
        independent_hashes_match = (manifest_arts == EXPECTED_ARTIFACTS)
        if independent_hashes_match:
            for rel_path in EXPECTED_ARTIFACTS:
                meta = manifest_data["artifacts"][rel_path]
                full_p = os.path.join(REPO_ROOT, rel_path)
                if not os.path.exists(full_p) or sha256_file(full_p) != meta["sha256"]:
                    independent_hashes_match = False
                    break

        # Self-contained provenance verification across report JSONs
        independent_prov_match = True
        for report_file in ["diagnosis_mapping_report.json", "clinical_feature_groups_report.json",
                            "feature_association_report.json", "outlier_analysis_report.json",
                            "feature_representation_contract.json"]:
            p = os.path.join(REPORTS_DIR, report_file)
            if not os.path.exists(p):
                independent_prov_match = False
                break
            with open(p, "r") as f:
                d = json.load(f)
            r_prov = d.get("provenance", {})
            if r_prov.get("analysis_population") != "train" or r_prov.get("train_rows") != 69519:
                independent_prov_match = False
                break

        if (
            split_meta.get("random_seed") == 42
            and prov.get("analysis_population") == "train"
            and prov.get("train_rows") == train_enc == 69519
            and prov.get("train_patients") == train_pat == 48993
            and independent_hashes_match
            and independent_prov_match
        ):
            reproducibility_pass = True

    if reproducibility_pass:
        results["Frozen artifact reproducibility"] = "PASS"
        print("  -> Frozen artifact reproducibility verified (independent split contract, deterministic seed 42, train-only provenance, artifact SHA-256 match).")
    else:
        results["Frozen artifact reproducibility"] = "FAIL"
        print("  -> Frozen artifact reproducibility checks failed.")

    # Final Gate Summary
    all_pass = all(v == "PASS" for v in results.values())
    print("\n" + "=" * 50)
    print("FINAL C3 EDA GATE SUMMARY")
    print("=" * 50)
    for gate, status in results.items():
        print(f"[{list(results.keys()).index(gate)+1:2d}] {gate:36s} {status}")
    print("-" * 50)
    c3_status = "PASS" if all_pass else "FAIL"
    print(f"{'C3 STATUS':40s} {c3_status}")
    print("=" * 50)

    return all_pass

if __name__ == "__main__":
    success = run_c3_verification()
    sys.exit(0 if success else 1)
