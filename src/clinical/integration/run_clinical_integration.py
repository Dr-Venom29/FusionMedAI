"""
Master Execution Pipeline for Clinical Phase C10: Integration & End-to-End Validation.
"""

import os
import sys
import json
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd

# Ensure project root in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.clinical.inference.service import ClinicalInferenceService
from src.clinical.inference.schema import ClinicalOutput
from src.clinical.inference.validator import ClinicalValidationError

EXPERIMENT_DIR = REPO_ROOT / "experiments" / "clinical" / "integration"


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_integration_pipeline():
    print("=" * 80)
    print("FUSIONMEDAI: CLINICAL PHASE C10 INTEGRATION & END-TO-END VALIDATION")
    print("=" * 80)

    # 1. Setup output paths
    exp_tables_dir = EXPERIMENT_DIR / "tables"
    exp_manifests_dir = EXPERIMENT_DIR / "manifests"
    exp_models_dir = EXPERIMENT_DIR / "models"

    for d in [exp_tables_dir, exp_manifests_dir, exp_models_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 2. Load dataset partitions
    data_dir = REPO_ROOT / "datasets" / "clinical" / "processed" / "splits"
    print(f"[1/8] Loading frozen splits from {data_dir}...")
    df_train = pd.read_csv(data_dir / "train.csv")
    df_val = pd.read_csv(data_dir / "val.csv")
    df_test = pd.read_csv(data_dir / "test.csv")

    # 3. Initialize Unified Inference Service
    print("[2/8] Initializing ClinicalInferenceService (Frozen C5-C9 stack)...")
    service = ClinicalInferenceService(
        train_df=df_train,
        val_df=df_val,
        models_cache_dir=exp_models_dir,
        operating_threshold=0.20,
        random_seed=42,
    )
    print("      Inference stack successfully initialized.")

    # 4. Deterministic Single-Patient Case Profiles Test
    print("[3/8] Testing deterministic single-encounter inference across 7 clinical profiles...")
    test_cases = [
        ("Low Risk Stable", df_test.iloc[0].to_dict()),
        ("False Negative Sub-Threshold", df_test.iloc[9].to_dict()),
        ("Low Risk Ambiguity", df_test.iloc[12].to_dict()),
        ("False Positive High Unc", df_test.iloc[68].to_dict()),
        ("Near-Threshold Ambiguity", df_test.iloc[119].to_dict()),
        ("High Risk Data Sparsity", df_test.iloc[981].to_dict()),
        ("Critical High Risk", df_test.iloc[11556].to_dict()),
    ]

    single_case_records = []
    for label, case_dict in test_cases:
        output = service.predict_encounter(case_dict)
        assert 0.0 <= output.probability <= 1.0
        assert 0.0 <= output.calibrated_probability <= 1.0
        assert output.uncertainty.std_probability >= 0.0
        assert len(output.feature_attributions) == 5
        single_case_records.append({
            "case_profile": label,
            "encounter_id": output.encounter_id,
            "raw_prob": output.probability,
            "calibrated_prob": output.calibrated_probability,
            "uncertainty_std": output.uncertainty.std_probability,
            "pi_95_lower": output.uncertainty.predictive_interval_95.lower,
            "pi_95_upper": output.uncertainty.predictive_interval_95.upper,
            "confidence": output.confidence,
            "decision_tier": output.decision_tier,
            "top_feature": output.feature_attributions[0].feature,
            "top_shap": output.feature_attributions[0].shap_value,
            "blind_spot_warning": output.shift_detection.blind_spot_warning,
        })
    df_single_cases = pd.DataFrame(single_case_records)
    df_single_cases.to_csv(exp_tables_dir / "single_patient_case_validation.csv", index=False)
    print(f"      Validated {len(test_cases)} deterministic patient profiles.")

    # 5. Test Schema-Valid Synthetic Encounter
    print("[4/8] Testing arbitrary schema-valid synthetic encounter (Profile B: Synthetic Encounter)...")
    synthetic_encounter = {
        "encounter_id": 99999901,
        "patient_nbr": 88888801,
        "race": "AfricanAmerican",
        "gender": "Female",
        "age": "[50-60)",
        "admission_type_id": 1,
        "discharge_disposition_id": 1,
        "admission_source_id": 7,
        "time_in_hospital": 4,
        "payer_code": "MC",
        "medical_specialty": "InternalMedicine",
        "num_lab_procedures": 45,
        "num_procedures": 1,
        "num_medications": 14,
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 1,
        "diag_1": "250.6",
        "diag_2": "401",
        "diag_3": "272",
        "number_diagnoses": 7,
        "max_glu_serum": "None",
        "A1Cresult": ">8",
        "metformin": "Steady",
        "insulin": "Steady",
        "change": "Ch",
        "diabetesMed": "Yes",
    }
    syn_out = service.predict_encounter(synthetic_encounter)
    assert isinstance(syn_out, ClinicalOutput)
    assert 0.0 <= syn_out.calibrated_probability <= 1.0
    print(f"      [Profile B - Synthetic Encounter]: p_raw={syn_out.probability:.4f}, p_cal={syn_out.calibrated_probability:.4f}, sigma_p={syn_out.uncertainty.std_probability:.4f}, tier={syn_out.decision_tier}.")

    # 6. Test Malformed Inputs & Controlled Validation Failures
    print("[5/8] Testing malformed input validation failure handling...")
    malformed_tests = [
        ("Empty dict", {}),
        ("Missing required column 'gender'", {"age": "[50-60)", "time_in_hospital": 3}),
        ("Negative medication count", {**synthetic_encounter, "num_medications": -5}),
        ("Zero hospital stay", {**synthetic_encounter, "time_in_hospital": 0}),
        ("Physiologically invalid age", {**synthetic_encounter, "age": "invalid_age_format"}),
        ("Unrecognized gender category", {**synthetic_encounter, "gender": "NonStandardCategory"}),
    ]

    handled_errors = 0
    for test_name, malformed_payload in malformed_tests:
        try:
            service.predict_encounter(malformed_payload)
            raise AssertionError(f"Expected ClinicalValidationError for {test_name}, but execution succeeded.")
        except ClinicalValidationError as e:
            handled_errors += 1

    assert handled_errors == len(malformed_tests), f"Expected {len(malformed_tests)} validation errors, caught {handled_errors}"
    print(f"      All {handled_errors} malformed inputs safely rejected with ClinicalValidationError.")

    # 7. Test TreeSHAP Additivity & C9 Blind-Spot Integration
    print("[6/8] Testing TreeSHAP additivity and C9 blind-spot detection...")
    # Exact TreeSHAP additivity: base_value (phi_0) + sum(phi_j) == model raw logit margin f(x)
    test_row = df_test.iloc[0:1]
    X_t, _, _ = service.preprocessor.transform(test_row)
    raw_margin = float(service.base_model.predict(X_t, prediction_type="RawFormulaVal")[0])
    shap_vals = service.explainer.shap_values(X_t)[0]
    base_val = float(service.explainer.expected_value)
    shap_sum_features = float(np.sum(shap_vals))
    shap_total = base_val + shap_sum_features
    abs_err = abs(shap_total - raw_margin)
    tol = 1e-6
    assert abs_err <= tol, f"TreeSHAP sum ({shap_total}) does not equal raw margin ({raw_margin}) within tol {tol}"
    print("      TreeSHAP Additivity Verification:")
    print(f"        base_value (phi_0)        = {base_val:+.6f}")
    print(f"        sum(SHAP) (sum_j phi_j)   = {shap_sum_features:+.6f}")
    print(f"        base_value + sum(SHAP)    = {shap_total:+.6f}")
    print(f"        model raw output f(x)     = {raw_margin:+.6f}")
    print(f"        absolute error            = {abs_err:.2e} (tolerance = {tol:.0e}) -> PASS")

    # Check Blind Spot Warning on Zero Inpatient
    zero_inp_encounter = {**synthetic_encounter, "number_inpatient": 0}
    out_zero = service.predict_encounter(zero_inp_encounter)
    if out_zero.calibrated_probability < 0.20 and out_zero.uncertainty.std_probability < service.unc_75th:
        assert out_zero.shift_detection.blind_spot_warning is True, "Expected blind spot warning for zero inpatient low-risk case"
        print("      C9 Blind-spot warning actively emitted for zero-inpatient profile.")

    # 8. Full End-to-End Batch Inference over Locked Test Partition (N=14,913)
    print(f"[7/8] Executing full end-to-end batch inference on locked test set (N={len(df_test):,})...")
    start_time = time.time()
    batch_outputs = service.predict_batch(df_test)
    elapsed_time = time.time() - start_time
    assert len(batch_outputs) == len(df_test), f"Expected {len(df_test)} outputs, got {len(batch_outputs)}"

    # Check output metrics integrity
    raw_probs = np.array([o.probability for o in batch_outputs])
    cal_probs = np.array([o.calibrated_probability for o in batch_outputs])
    std_probs = np.array([o.uncertainty.std_probability for o in batch_outputs])
    preds = np.array([o.prediction for o in batch_outputs])

    assert (raw_probs >= 0.0).all() and (raw_probs <= 1.0).all()
    assert (cal_probs >= 0.0).all() and (cal_probs <= 1.0).all()
    assert (std_probs >= 0.0).all()

    throughput = len(df_test) / elapsed_time

    batch_summary = [
        {"metric": "Total Encounters Processed", "value": f"{len(batch_outputs):,}"},
        {"metric": "Batch Execution Time (seconds)", "value": f"{elapsed_time:.2f} s"},
        {"metric": "Local CPU Batch Inference Throughput", "value": f"{throughput:.1f} encounters/sec"},
        {"metric": "Mean Raw Probability", "value": f"{np.mean(raw_probs):.4f}"},
        {"metric": "Mean Calibrated Probability", "value": f"{np.mean(cal_probs):.4f}"},
        {"metric": "Mean Predictive Uncertainty (std)", "value": f"{np.mean(std_probs):.4f}"},
        {"metric": "Median Predictive Uncertainty", "value": f"{np.median(std_probs):.4f}"},
        {"metric": "Positive Flag Rate (theta=0.20)", "value": f"{np.mean(preds)*100:.2f}%"},
        {"metric": "Zero Crashes / Schema Errors", "value": "Verified (100% compliant)"},
        {"metric": "Technical Scope Note", "value": "Technical end-to-end integration benchmark; not external clinical validation"},
    ]
    df_batch_summary = pd.DataFrame(batch_summary)
    df_batch_summary.to_csv(exp_tables_dir / "batch_inference_summary.csv", index=False)
    print(f"      Batch inference complete in {elapsed_time:.2f}s ({throughput:.1f} encounters/sec local CPU throughput).")

    # 9. Cryptographic SHA-256 Manifest
    print("[8/8] Generating cryptographic SHA-256 manifest linking frozen upstream C5-C9 artifacts...")
    upstream_manifests = {
        "c5_catboost_hpo": REPO_ROOT / "experiments" / "clinical" / "catboost_hpo" / "manifests" / "benchmarking_manifest.json",
        "c6_explainability": REPO_ROOT / "experiments" / "clinical" / "explainability" / "manifests" / "c6_explainability_manifest.json",
        "c7_calibration": REPO_ROOT / "experiments" / "clinical" / "calibration" / "manifests" / "c7_calibration_manifest.json",
        "c8_uncertainty": REPO_ROOT / "experiments" / "clinical" / "uncertainty" / "manifests" / "c8_uncertainty_manifest.json",
        "c9_robustness": REPO_ROOT / "experiments" / "clinical" / "robustness" / "manifests" / "c9_robustness_manifest.json",
    }
    
    upstream_hashes = {}
    for phase_key, manifest_p in upstream_manifests.items():
        if manifest_p.exists():
            upstream_hashes[phase_key] = {
                "manifest_path": str(manifest_p.relative_to(REPO_ROOT)).replace("\\", "/"),
                "sha256": compute_sha256(manifest_p),
            }

    data_splits = {
        "train_csv": data_dir / "train.csv",
        "val_csv": data_dir / "val.csv",
        "test_csv": data_dir / "test.csv",
    }
    data_hashes = {}
    for split_key, split_p in data_splits.items():
        if split_p.exists():
            data_hashes[split_key] = {
                "path": str(split_p.relative_to(REPO_ROOT)).replace("\\", "/"),
                "sha256": compute_sha256(split_p),
            }

    manifest = {
        "experiment_phase": "Clinical Phase C10: Clinical Integration & End-to-End Validation",
        "scope_declaration": "Technical end-to-end software integration and internal output verification; not external/prospective clinical validation",
        "frozen_components": {
            "preprocessor": "ClinicalPreprocessor (D=119)",
            "base_model": "CatBoost_HPO_Tuned_Candidate",
            "calibrator": "Isotonic_Regression",
            "uncertainty_ensemble": "BootstrapCatBoostEnsemble (M=50)",
            "explainer": "Exact TreeSHAP (Additive Margins)",
            "shift_safeguards": "Prior-Inpatient Blind Spot & Missingness Detector",
        },
        "operating_threshold": 0.20,
        "test_encounters_verified": len(df_test),
        "local_cpu_batch_throughput_enc_per_sec": float(f"{throughput:.1f}"),
        "upstream_phase_manifests": upstream_hashes,
        "canonical_data_splits": data_hashes,
        "artifacts": {},
    }

    for path in sorted(list(exp_tables_dir.glob("*.csv"))):
        rel_str = str(path.relative_to(EXPERIMENT_DIR)).replace("\\", "/")
        manifest["artifacts"][rel_str] = {
            "sha256": compute_sha256(path),
            "size_bytes": path.stat().st_size,
        }

    manifest_path = exp_manifests_dir / "c10_integration_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 80)
    print("PHASE C10 EXECUTION COMPLETE. ALL INTEGRATION TESTS PASSED.")
    print("=" * 80)


if __name__ == "__main__":
    run_integration_pipeline()
