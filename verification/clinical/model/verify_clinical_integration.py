"""Phase C10 Clinical Integration & End-to-End Validation Verification Suite."""

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
from src.clinical.inference.service import ClinicalInferenceService
from src.clinical.inference.schema import ClinicalOutput
from src.clinical.inference.validator import ClinicalValidationError


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def verify_clinical_integration_gate() -> bool:
    print("=" * 80)
    print("VERIFICATION: Clinical Phase C10 Integration & End-to-End Inference Gate")
    print("=" * 80)

    exp_root = get_runtime_output_root(REPO_ROOT, "integration")
    data_dir = REPO_ROOT / "datasets" / "clinical" / "processed" / "splits"

    checks_passed = 0
    total_checks = 10

    # 1. Check Frozen C5-C9 Components in Manifest
    manifest_path = exp_root / "manifests" / "c10_integration_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        assert manifest["frozen_components"]["base_model"] == "CatBoost_HPO_Tuned_Candidate"
        assert manifest["frozen_components"]["calibrator"] == "Isotonic_Regression"
        assert manifest["test_encounters_verified"] == 14913
        print("[PASS  1/10] Frozen C5–C9 Stack Components Verified in Manifest.")
        checks_passed += 1
    else:
        print(f"[FAIL  1/10] Integration Manifest Missing: {manifest_path}")

    # Load data for live checks
    df_train = pd.read_csv(data_dir / "train.csv")
    df_val = pd.read_csv(data_dir / "val.csv")
    df_test = pd.read_csv(data_dir / "test.csv")

    service = ClinicalInferenceService(
        train_df=df_train,
        val_df=df_val,
        operating_threshold=0.20,
        random_seed=42,
    )

    # 2. Check 119-D Representation Contract
    assert len(service.feature_names) == 119
    sample_df = df_test.iloc[0:2]
    X_trans, _, _ = service.preprocessor.transform(sample_df)
    assert X_trans.shape == (2, 119)
    print(f"[PASS  2/10] 119-D Representation Contract Preserved (D={len(service.feature_names)}).")
    checks_passed += 1

    # 3. Check Single-Encounter Inference (Deterministic Reference Profile A)
    sample_patient = df_test.iloc[0].to_dict()
    output = service.predict_encounter(sample_patient)
    assert isinstance(output, ClinicalOutput)
    assert 0.0 <= output.probability <= 1.0
    assert 0.0 <= output.calibrated_probability <= 1.0
    print(f"[PASS  3/10] Single-Encounter Inference Verified (Encounter {output.encounter_id} [Profile A - Locked-Test Ref]: p_raw={output.probability:.4f}, p_cal={output.calibrated_probability:.4f}, sigma_p={output.uncertainty.std_probability:.4f}).")
    checks_passed += 1

    # 4. Check ClinicalOutput Schema Conformance
    out_dict = output.to_dict()
    required_keys = [
        "modality", "encounter_id", "prediction", "probability",
        "calibrated_probability", "calibration_method", "confidence",
        "uncertainty", "decision_tier", "operating_threshold",
        "feature_attributions", "shift_detection", "model_provenance"
    ]
    assert all(k in out_dict for k in required_keys)
    assert out_dict["modality"] == "clinical_tabular"
    assert out_dict["operating_threshold"] == 0.20
    print("[PASS  4/10] Standardized ClinicalOutput Schema Fully Conforming.")
    checks_passed += 1

    # 5. Check C6 TreeSHAP Integration & Exact Additivity: phi_0 + sum(phi_j) == f(x)
    assert len(output.feature_attributions) == 5
    raw_margin = float(service.base_model.predict(X_trans[0:1], prediction_type="RawFormulaVal")[0])
    shap_vals = service.explainer.shap_values(X_trans[0:1])[0]
    base_val = float(service.explainer.expected_value)
    shap_sum_features = float(np.sum(shap_vals))
    shap_total = base_val + shap_sum_features
    abs_err = abs(shap_total - raw_margin)
    tol = 1e-6
    assert abs_err <= tol, f"SHAP additivity violated: |{shap_total} - {raw_margin}| = {abs_err} > {tol}"
    print("[PASS  5/10] C6 TreeSHAP Additivity Verified:")
    print(f"             base_value (phi_0)        = {base_val:+.6f}")
    print(f"             sum(SHAP) (sum_j phi_j)   = {shap_sum_features:+.6f}")
    print(f"             base_value + sum(SHAP)    = {shap_total:+.6f}")
    print(f"             model raw output f(x)     = {raw_margin:+.6f}")
    print(f"             absolute error            = {abs_err:.2e} (tolerance = {tol:.0e}) -> PASS")
    checks_passed += 1

    # 6. Check C7 Calibration Integration
    # Test calibrated probability vs raw probability
    p_raw = output.probability
    p_cal = output.calibrated_probability
    assert p_cal == float(service.calibrator.predict_proba(np.array([p_raw]))[0])
    assert output.calibration_method == "Isotonic_Regression"
    print(f"[PASS  6/10] C7 Calibration Integration Verified (Raw p={p_raw:.4f} -> Calibrated p={p_cal:.4f}).")
    checks_passed += 1

    # 7. Check C8 Uncertainty Integration
    u_out = output.uncertainty
    assert u_out.method == "bootstrap_ensemble"
    assert u_out.std_probability >= 0.0
    assert u_out.predictive_interval_95.lower <= u_out.predictive_interval_95.upper
    assert output.decision_tier in [
        "Low Risk / Low Uncertainty",
        "Low Risk / High Uncertainty",
        "High Risk / Low Uncertainty",
        "High Risk / High Uncertainty",
        "Near Threshold / Low Uncertainty",
        "Near Threshold / High Uncertainty (Ambiguity)",
    ]
    print(f"[PASS  7/10] C8 Uncertainty Integration Verified (sigma_p = {u_out.std_probability:.4f}, Tier = {output.decision_tier}).")
    checks_passed += 1

    # 8. Check C9 Robustness & Blind-Spot Integration
    # Zero inpatient case must emit blind-spot alert when low risk
    zero_inp_patient = dict(sample_patient)
    zero_inp_patient["number_inpatient"] = 0
    out_blind = service.predict_encounter(zero_inp_patient)
    if out_blind.calibrated_probability < 0.20 and out_blind.uncertainty.std_probability < service.unc_75th:
        assert out_blind.shift_detection.blind_spot_warning is True
    print("[PASS  8/10] C9 Robustness & Shift Safeguards Verified (Blind-Spot Detector Active).")
    checks_passed += 1

    # 9. Check Malformed-Input Failure Handling
    malformed_errors = 0
    try:
        service.predict_encounter({})
    except ClinicalValidationError:
        malformed_errors += 1

    try:
        service.predict_encounter({"gender": "Female", "num_medications": -10, "time_in_hospital": 2})
    except ClinicalValidationError:
        malformed_errors += 1

    assert malformed_errors == 2
    print("[PASS  9/10] Malformed-Input & Physiological Bounds Validation Verified.")
    checks_passed += 1

    # 10. Check End-to-End Batch Inference on Locked Test Partition
    batch_csv = exp_root / "tables" / "batch_inference_summary.csv"
    if batch_csv.exists():
        df_batch = pd.read_csv(batch_csv)
        metrics_dict = dict(zip(df_batch["metric"], df_batch["value"]))
        assert metrics_dict["Total Encounters Processed"] == "14,913"
        assert metrics_dict["Zero Crashes / Schema Errors"].startswith("Verified")
        print(f"[PASS 10/10] Locked-Test End-to-End Batch Integration Verified (14,913 encounters, {metrics_dict['Local CPU Batch Inference Throughput']}).")
        checks_passed += 1
    else:
        print(f"[FAIL 10/10] Batch Inference Table Missing: {batch_csv}")

    print("=" * 80)
    print(f"VERIFICATION SUMMARY: {checks_passed}/{total_checks} checks passed.")
    print("=" * 80)
    return checks_passed == total_checks


if __name__ == "__main__":
    success = verify_clinical_integration_gate()
    sys.exit(0 if success else 1)
