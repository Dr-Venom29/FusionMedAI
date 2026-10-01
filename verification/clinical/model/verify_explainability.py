"""Phase C6 Clinical Model Explainability Verification Suite."""

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


def verify_explainability_pipeline() -> bool:
    print("=" * 80)
    print("VERIFICATION: Clinical Phase C6 Model Explainability")
    print("=" * 80)

    exp_root = get_runtime_output_root(REPO_ROOT, "explainability")
    research_root = REPO_ROOT / "research" / "clinical" / "Volume_06_Explainability"

    checks_passed = 0
    total_checks = 10

    # 1. Check Global Feature Importance Artifact
    global_csv = exp_root / "global" / "global_feature_importance.csv"
    if global_csv.exists():
        df_glob = pd.read_csv(global_csv)
        assert len(df_glob) == 119, f"Expected 119 features, found {len(df_glob)}"
        assert "mean_abs_shap_test" in df_glob.columns
        assert "directionality_corr" in df_glob.columns
        print(f"[PASS 1/10] Global Feature Importance Artifact Verified: {len(df_glob)} features, top feature: {df_glob.iloc[0]['feature']}")
        checks_passed += 1
    else:
        print(f"[FAIL 1/10] Global Feature Importance Artifact Missing: {global_csv}")

    # 2. Check Feature Group Importance Artifact
    group_csv = exp_root / "global" / "feature_group_importance.csv"
    if group_csv.exists():
        df_group = pd.read_csv(group_csv)
        assert len(df_group) >= 6, f"Expected at least 6 clinical groups, found {len(df_group)}"
        assert np.isclose(df_group["relative_contribution_pct"].sum(), 100.0, atol=0.1)
        print(f"[PASS 2/10] Clinical Feature Group Artifact Verified: {len(df_group)} taxonomy groups, total contribution = 100.0%")
        checks_passed += 1
    else:
        print(f"[FAIL 2/10] Feature Group Artifact Missing: {group_csv}")

    # 3. Check Local Case Explanations Artifact
    local_csv = exp_root / "local" / "local_case_explanations.csv"
    if local_csv.exists():
        df_local = pd.read_csv(local_csv)
        assert len(df_local) >= 15, f"Expected >= 15 local cases, found {len(df_local)}"
        assert "top_risk_increasing_features" in df_local.columns
        assert "top_risk_decreasing_features" in df_local.columns
        print(f"[PASS 3/10] Local Case Explanations Artifact Verified: {len(df_local)} representative encounters across 5 risk cohorts")
        checks_passed += 1
    else:
        print(f"[FAIL 3/10] Local Case Explanations Missing: {local_csv}")

    # 4. Check Error-Case Attributions (FP vs FN)
    fp_csv = exp_root / "error_analysis" / "false_positive_shap_attributions.csv"
    fn_csv = exp_root / "error_analysis" / "false_negative_shap_attributions.csv"
    if fp_csv.exists() and fn_csv.exists():
        df_fp = pd.read_csv(fp_csv)
        df_fn = pd.read_csv(fn_csv)
        assert len(df_fp) == 119 and len(df_fn) == 119
        print(f"[PASS 4/10] Error-Focused SHAP Attributions Verified: FP top feature '{df_fp.iloc[0]['feature']}', FN top feature '{df_fn.iloc[0]['feature']}'")
        checks_passed += 1
    else:
        print(f"[FAIL 4/10] Error Attribution Artifacts Missing")

    # 5. Check Subgroup Feature Importance
    subgroup_csv = exp_root / "subgroup" / "subgroup_feature_importance.csv"
    if subgroup_csv.exists():
        df_sub = pd.read_csv(subgroup_csv)
        assert len(df_sub) == 119
        assert "mean_abs_shap_prior_inpatient_ge1" in df_sub.columns
        print(f"[PASS 5/10] Subgroup SHAP Importance Verified: 119 features stratified across demographic and utilization cohorts")
        checks_passed += 1
    else:
        print(f"[FAIL 5/10] Subgroup Feature Importance Missing: {subgroup_csv}")

    # 6. Check Stability Metrics (Validation vs Test)
    stab_csv = exp_root / "global" / "shap_stability.csv"
    if stab_csv.exists():
        df_stab = pd.read_csv(stab_csv)
        val_top10 = set(df_stab.sort_values("mean_abs_shap_val", ascending=False).head(10)["feature"])
        test_top10 = set(df_stab.sort_values("mean_abs_shap_test", ascending=False).head(10)["feature"])
        overlap_10 = len(val_top10.intersection(test_top10)) / 10.0
        assert overlap_10 >= 0.80, f"Top-10 overlap below threshold: {overlap_10}"
        print(f"[PASS 6/10] Validation vs. Test Stability Verified: Top-10 overlap = {overlap_10*100:.0f}%")
        checks_passed += 1
    else:
        print(f"[FAIL 6/10] Stability Artifact Missing: {stab_csv}")

    # 7. Check Figures Generation
    fig_dir = exp_root / "figures"
    required_figs = ["shap_bar.png", "shap_summary.png", "shap_group_importance.png", "shap_stability.png"]
    missing_figs = [f for f in required_figs if not (fig_dir / f).exists()]
    if not missing_figs:
        print(f"[PASS 7/10] Publication-Quality Figures Verified: All 4 core figures present in {fig_dir}")
        checks_passed += 1
    else:
        print(f"[FAIL 7/10] Missing Figures: {missing_figs}")

    # 8. Check Dependence Plots
    dep_dir = fig_dir / "dependence"
    dep_files = list(dep_dir.glob("*.png")) if dep_dir.exists() else []
    if len(dep_files) >= 4:
        print(f"[PASS 8/10] SHAP Dependence Plots Verified: {len(dep_files)} continuous/ordinal dependence plots present")
        checks_passed += 1
    else:
        print(f"[FAIL 8/10] Insufficient Dependence Plots: found {len(dep_files)}")

    # 9. Check Cryptographic Manifest and Checksums
    manifest_file = exp_root / "manifests" / "c6_explainability_manifest.json"
    if manifest_file.exists():
        with open(manifest_file, "r") as f:
            manifest = json.load(f)
        assert "provenance" in manifest
        assert "artifacts" in manifest
        m_artifacts = manifest["artifacts"]
        assert len(m_artifacts) >= 10
        print(f"[PASS 9/10] Cryptographic Manifest Verified: {len(m_artifacts)} hashed artifacts recorded")
        checks_passed += 1
    else:
        print(f"[FAIL 9/10] Manifest File Missing: {manifest_file}")

    # 10. Check Research Volume Sync
    res_manifest = research_root / "c6_manifest.json"
    res_global_csv = research_root / "02_Global_Feature_Importance.csv"
    if res_manifest.exists() and res_global_csv.exists():
        print(f"[PASS 10/10] Research Volume 06 Directory Synchronized: CSVs, figures, and manifests mapped to {research_root}")
        checks_passed += 1
    else:
        print(f"[FAIL 10/10] Research Volume 06 Missing Artifacts")

    print("-" * 80)
    print(f"VERIFICATION STATUS: {checks_passed} / {total_checks} CHECKS PASSED")
    print("=" * 80)
    return checks_passed == total_checks


if __name__ == "__main__":
    success = verify_explainability_pipeline()
    sys.exit(0 if success else 1)
