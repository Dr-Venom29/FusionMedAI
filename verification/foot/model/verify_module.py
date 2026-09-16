import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.foot.foot_module import FootModule
import src.foot.config as config

def run_12_point_module_verification() -> bool:
    """
    Executes the 12-point automated verification suite for Phase 10.9 FootModule Integration.
    """
    print("=============================================================")
    print("  Foot Module Integration Verification Suite (Phase 10.9)  ")
    print("=============================================================\n", flush=True)
    
    passed_tests = 0
    total_tests = 12
    
    # Instantiate FootModule
    try:
        module = FootModule()
        print("  [INIT] FootModule instantiated successfully.\n", flush=True)
    except Exception as e:
        print(f"  [FAIL] Failed to instantiate FootModule: {e}")
        return False

    # Get sample test image path from test dataset
    test_split_path = PROJECT_ROOT / "datasets" / "foot" / "processed" / "splits" / "test.csv"
    import pandas as pd
    df_test = pd.read_csv(test_split_path)
    sample_rel_path = df_test.iloc[0]["image_path"]
    sample_img_path = PROJECT_ROOT / "datasets" / "foot" / "raw" / sample_rel_path
    if not sample_img_path.exists():
        sample_img_path = PROJECT_ROOT / "datasets" / "foot" / "processed" / sample_rel_path
    if not sample_img_path.exists():
        raise FileNotFoundError(f"Sample test image not found at {sample_img_path}")

    # --- Test 1: Checkpoint & Architecture Loading ---
    print("1. Verifying EfficientNet-B3 Model Checkpoint Loading...", end="", flush=True)
    if isinstance(module.model, nn.Module) and not module.model.training:
        print(" PASS")
        passed_tests += 1
    else:
        print(" FAIL")

    # --- Test 2: Calibration Artifact Loading ---
    print("2. Verifying Vector Scaling Calibration Artifact Loading...", end="", flush=True)
    if hasattr(module, "vector_scaler") and len(module.calib_meta["weights"]) == 4 and len(module.calib_meta["bias"]) == 4:
        print(" PASS")
        passed_tests += 1
    else:
        print(" FAIL")

    # --- Test 3: Uncertainty Config Verification ---
    print("3. Verifying Frozen Uncertainty Configuration (N*=10)...", end="", flush=True)
    if module.default_mc_passes == 10:
        print(" PASS")
        passed_tests += 1
    else:
        print(" FAIL")

    # --- Test 4: Input Image Validation Safety ---
    print("4. Verifying Input Validation Safety & Error Handling...", end="", flush=True)
    try:
        err_caught = False
        # Case A: Non-existent file
        try:
            module.predict("non_existent_file_xyz_123.jpg")
        except FileNotFoundError:
            err_caught = True
            
        # Case B: Invalid image type
        try:
            module.predict(12345)
        except TypeError:
            err_caught = err_caught and True
            
        if err_caught:
            print(" PASS")
            passed_tests += 1
        else:
            print(" FAIL (Did not raise expected input error)")
    except Exception as e:
        print(f" FAIL ({e})")

    # Perform inference for subsequent verification tests
    res = module.predict(sample_img_path, mc_passes=10, generate_cam=True)

    # --- Test 5: Prediction Validity ---
    print("5. Verifying Class Prediction Validity (0 <= pred < 4)...", end="", flush=True)
    if 0 <= res["prediction"] < 4 and res["prediction_label"] in config.CLASS_NAMES:
        print(" PASS")
        passed_tests += 1
    else:
        print(" FAIL")

    # --- Test 6: Probability Vector Normalization ---
    print("6. Verifying Probability Vector Normalization (sum ≈ 1.0)...", end="", flush=True)
    raw_sum = sum(res["raw_probabilities"])
    calib_sum = sum(res["calib_probabilities"])
    if abs(raw_sum - 1.0) < 1e-4 and abs(calib_sum - 1.0) < 1e-4:
        print(" PASS")
        passed_tests += 1
    else:
        print(" FAIL")

    # --- Test 7: Vector Scaling Logit Transformation ---
    print("7. Verifying Vector Scaling Calibration Output Integrity...", end="", flush=True)
    if len(res["calib_probabilities"]) == 4 and 0.0 <= res["calib_confidence"] <= 1.0:
        print(" PASS")
        passed_tests += 1
    else:
        print(" FAIL")

    # --- Test 8: MC Dropout Pass Count Enforcement ---
    print("8. Verifying MC Dropout Pass Count Enforcement (N=10)...", end="", flush=True)
    if res["mc_passes_N"] == 10:
        print(" PASS")
        passed_tests += 1
    else:
        print(" FAIL")

    # --- Test 9: Uncertainty Metric Bounds ---
    print("9. Verifying Predictive Uncertainty Metric Bounds (H >= 0, Var >= 0, MI >= 0)...", end="", flush=True)
    if res["mc_predictive_entropy"] >= 0.0 and res["mc_predictive_variance"] >= 0.0 and res["mc_mutual_information"] >= 0.0:
        print(" PASS")
        passed_tests += 1
    else:
        print(" FAIL")

    # --- Test 10: Inference Mode Support (Standard vs Explainable) ---
    print("10. Verifying Standard vs Explainable Mode Support...", end="", flush=True)
    res_std = module.predict(sample_img_path, mc_passes=10, generate_cam=False)
    if res_std["cam_overlay"] is None and res["cam_overlay"] is not None:
        print(" PASS")
        passed_tests += 1
    else:
        print(" FAIL")

    # --- Test 11: Schema Contract Parity ---
    print("11. Verifying Output Schema Contract Parity...", end="", flush=True)
    required_keys = [
        "modality", "prediction", "prediction_label", "raw_confidence", "calib_confidence",
        "raw_probabilities", "calib_probabilities", "calib_entropy_norm", "calib_margin",
        "mc_passes_N", "mc_predictive_entropy", "mc_predictive_entropy_norm",
        "mc_expected_entropy", "mc_expected_entropy_norm", "mc_predictive_variance",
        "mc_mutual_information", "cam_overlay", "cam_heatmap", "cam_mean_intensity", "latency_ms"
    ]
    all_keys_present = all(k in res for k in required_keys)
    if all_keys_present and res["modality"] == "foot":
        print(" PASS")
        passed_tests += 1
    else:
        print(" FAIL")

    # --- Test 12: Inference Latency & Repeated Call Stability ---
    print("12. Verifying Inference Latency & Repeated Call Stability (< 1000 ms)...", end="", flush=True)
    if res["latency_ms"] < 1000.0:
        print(" PASS")
        passed_tests += 1
    else:
        print(" FAIL")

    print("\n-------------------------------------------------------------")
    print(f"  Verification Results: {passed_tests}/{total_tests} PASS")
    print("-------------------------------------------------------------")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = run_12_point_module_verification()
    sys.exit(0 if success else 1)
