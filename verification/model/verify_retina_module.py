import os
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import src.config as config
from src.retina_module import RetinaModule

def verify_retina_module_acceptance():
    """
    Step 9 - Final End-to-End Acceptance Test
    Validates the entire integrated RetinaModule contract:
    Inference -> Calibration -> MC Dropout -> Grad-CAM -> Unified Output
    """
    print("\n==================================================")
    print("Verification: Step 9 Retina Module Acceptance Test")
    print("==================================================\n")

    # ----------------------------------------------------
    # 1. Verify required files exist
    # ----------------------------------------------------
    print("[1/9] Verifying file presence...")
    checkpoint_path = Path(config.PROJECT_ROOT) / "experiments" / "efficientnet_b3" / "checkpoints" / "best_model.pt"
    required_files = [
        ("Retina Module source", Path(config.PROJECT_ROOT) / "src" / "retina_module.py"),
        ("Verification script", Path(config.PROJECT_ROOT) / "verification" / "model" / "verify_retina_module.py"),
        ("EfficientNet-B3 Checkpoint", checkpoint_path),
        ("Step 8 Predictions CSV", Path(config.PROJECT_ROOT) / "results" / "uncertainty" / "predictions.csv"),
        ("Step 8 Summary Metrics JSON", Path(config.PROJECT_ROOT) / "results" / "uncertainty" / "uncertainty_metrics.json"),
        ("Step 8 MC Validation JSON", Path(config.PROJECT_ROOT) / "results" / "uncertainty" / "mc_dropout_validation.json")
    ]
    
    for label, filepath in required_files:
        assert filepath.exists(), f"Required file '{label}' not found at: {filepath}"
        print(f"  [OK] Found {label} ({filepath.name})")

    # Verify uncertainty experiments directory
    exp_uncertainty_dir = Path(config.PROJECT_ROOT) / "experiments" / "uncertainty"
    assert exp_uncertainty_dir.exists(), "Uncertainty experiments folder missing."
    print(f"  [OK] Found uncertainty experiments directory: {exp_uncertainty_dir.name}")
    print("  File verification completed successfully.")

    # ----------------------------------------------------
    # 2. Instantiate Integrated RetinaModule
    # ----------------------------------------------------
    print("\n[2/9] Instantiating integrated RetinaModule...")
    try:
        module = RetinaModule(
            checkpoint_path=checkpoint_path,
            device="cpu"  # Keep CPU-based for acceptance test
        )
        print("  [OK] RetinaModule instantiated successfully.")
    except Exception as e:
        print(f"  [FAIL] Failed to instantiate RetinaModule: {e}")
        sys.exit(1)

    # Assert architecture properties
    assert module.model.__class__.__name__ == "EfficientNetB3", f"Model backbone is {module.model.__class__.__name__}, not EfficientNetB3."
    assert config.MODEL_NAME == "efficientnet_b3", f"Expected model name efficientnet_b3, got {config.MODEL_NAME}"
    print(f"  [OK] Checkpoint verified: {checkpoint_path.name} (Architecture: EfficientNet-B3)")

    # ----------------------------------------------------
    # 3. Verify Calibration dynamically loads temperature
    # ----------------------------------------------------
    print("\n[3/9] Verifying calibration temperature...")
    print(f"  Dynamically loaded Temperature T = {module.temperature:.4f}")
    assert np.isclose(module.temperature, 1.6218, atol=1e-3), f"Loaded temperature {module.temperature:.4f} is far from expected 1.6218."
    print("  [OK] Calibration temperature validated.")

    # ----------------------------------------------------
    # 4. Multi-Image Acceptance & State Isolation Test (3 Images)
    # ----------------------------------------------------
    print("\n[4/9] Executing multi-image test (3 images) for state isolation...")
    test_images_dir = Path(config.PROJECT_ROOT) / "datasets" / "raw" / "aptos2019" / "train_images"
    sample_images = sorted(list(test_images_dir.glob("*.png")))
    
    assert len(sample_images) >= 3, f"At least 3 images required in {test_images_dir} for multi-image testing."
    test_subset = sample_images[:3]
    
    mc_passes = 5 # 5 passes for the acceptance run
    results = []
    
    for idx, img_path in enumerate(test_subset):
        print(f"  Predicting Image {idx+1}: {img_path.name}...")
        res = module.predict(image=img_path, mc_passes=mc_passes, generate_cam=True)
        results.append((img_path, res))
        print(f"    - Predicted: Class {res['prediction']} ({res['prediction_label']})")
        print(f"    - Calibrated Confidence: {res['calib_confidence']:.2%}")
        print(f"    - MC Predictive Entropy (Norm): {res['mc_predictive_entropy_norm']:.4f}")
        print(f"    - MC Mutual Information: {res['mc_mutual_information']:.6f}")
        print(f"    - Latency: {res['latency_ms']:.1f} ms")
        
    # Verify predictions are not identical due to state leaks
    print("  Asserting that images are processed independently and do not leak state...")
    # Compare raw probabilities of consecutive images to confirm state separation
    prob1 = np.array(results[0][1]["calib_probabilities"])
    prob2 = np.array(results[1][1]["calib_probabilities"])
    assert not np.allclose(prob1, prob2), "State leak detected: adjacent predictions have identical probabilities."
    print("  [OK] State isolation verified.")

    # ----------------------------------------------------
    # 5. Verify Unified Output Contract (Schema check)
    # ----------------------------------------------------
    print("\n[5/9] Verifying unified output contract schema...")
    res = results[0][1] # Use results of the first image for detailed checks
    
    expected_structure = {
        "prediction": int,
        "prediction_label": str,
        "raw_confidence": float,
        "calib_confidence": float,
        "raw_probabilities": list,
        "calib_probabilities": list,
        "calib_entropy_norm": float,
        "calib_margin": float,
        "mc_predictive_entropy": float,
        "mc_predictive_entropy_norm": float,
        "mc_expected_entropy": float,
        "mc_expected_entropy_norm": float,
        "mc_predictive_variance": float,
        "mc_mutual_information": float,
        "cam_overlay": np.ndarray,
        "cam_heatmap": np.ndarray,
        "cam_mean_intensity": float,
        "latency_ms": float
    }
    
    for key, expected_type in expected_structure.items():
        assert key in res, f"Unified output missing expected key: {key}"
        assert isinstance(res[key], expected_type), f"Key '{key}' expected type {expected_type}, got {type(res[key])}"
        assert res[key] is not None, f"Key '{key}' is None (auxiliary branch failed silently)."
        
    print("  [OK] Unified contract schema matches specification.")

    # ----------------------------------------------------
    # 6. Verify Probability distributions sum to 1
    # ----------------------------------------------------
    print("\n[6/9] Verifying probability distribution constraints...")
    raw_probs = np.array(res["raw_probabilities"])
    calib_probs = np.array(res["calib_probabilities"])
    
    assert np.all(raw_probs >= 0.0) and np.all(raw_probs <= 1.0), "Raw probabilities out of range."
    assert np.all(calib_probs >= 0.0) and np.all(calib_probs <= 1.0), "Calibrated probabilities out of range."
    
    raw_sum = float(np.sum(raw_probs))
    calib_sum = float(np.sum(calib_probs))
    
    assert np.isclose(raw_sum, 1.0), f"Raw probabilities sum to {raw_sum} instead of 1.0"
    assert np.isclose(calib_sum, 1.0), f"Calibrated probabilities sum to {calib_sum} instead of 1.0"
    print(f"  Raw sum:        {raw_sum:.6f}")
    print(f"  Calibrated sum: {calib_sum:.6f}")
    print("  [OK] Probabilities sum to 1.0.")

    # ----------------------------------------------------
    # 7. Verify MC Dropout Stochasticity Variation
    # ----------------------------------------------------
    print("\n[7/9] Verifying MC Dropout stochasticity variation...")
    # To check that Pass 1 != Pass 2, we will invoke predict on the image
    # and verify that the internal forward passes are indeed producing variance.
    # We can do a 2-pass test and check if intermediate logits/probs differ.
    
    # We do a predict call with mc_passes=5
    # The output has mc_predictive_variance > 0
    # Let's verify predictive variance is non-zero
    assert res["mc_predictive_variance"] > 0, "Zero variance detected. Dropout layer failed to apply stochasticity."
    print(f"  MC Predictive Variance: {res['mc_predictive_variance']:.6f} (> 0)")
    print("  [OK] MC Dropout stochasticity variation confirmed.")

    # ----------------------------------------------------
    # 8. Verify Uncertainty Metrics
    # ----------------------------------------------------
    print("\n[8/9] Verifying numerical bounds of uncertainty metrics...")
    assert np.isfinite(res["mc_predictive_entropy"]), "Predictive entropy is infinite."
    assert np.isfinite(res["mc_expected_entropy"]), "Expected entropy is infinite."
    assert np.isfinite(res["mc_predictive_variance"]), "Predictive variance is infinite."
    assert np.isfinite(res["mc_mutual_information"]), "Mutual information is infinite."
    
    assert res["mc_predictive_entropy"] >= 0, "Predictive entropy is negative."
    assert res["mc_expected_entropy"] >= 0, "Expected entropy is negative."
    assert res["mc_predictive_variance"] >= 0, "Predictive variance is negative."
    # Allow minor float precision tolerance
    assert res["mc_mutual_information"] >= -1e-6, f"Mutual information is significantly negative: {res['mc_mutual_information']}"
    
    print(f"  MC Predictive Entropy (Norm): {res['mc_predictive_entropy_norm']:.4f}")
    print(f"  MC Expected Entropy (Norm):   {res['mc_expected_entropy_norm']:.4f}")
    print(f"  MC Mutual Information:        {res['mc_mutual_information']:.6f}")
    print("  [OK] Uncertainty metrics verified within correct numerical bounds.")

    # ----------------------------------------------------
    # 9. Verify Grad-CAM Visualization overlay
    # ----------------------------------------------------
    print("\n[9/9] Verifying Grad-CAM heatmap visual overlay...")
    overlay = res["cam_overlay"]
    heatmap = res["cam_heatmap"]
    
    assert overlay.shape[2] == 3, f"Expected 3 color channels, got shape {overlay.shape}"
    assert heatmap.shape[2] == 3, f"Expected 3 color channels, got shape {heatmap.shape}"
    assert overlay.dtype == np.uint8, "Expected uint8 overlay."
    assert heatmap.dtype == np.uint8, "Expected uint8 heatmap."
    assert 0.0 <= res["cam_mean_intensity"] <= 1.0, f"Grad-CAM mean intensity {res['cam_mean_intensity']:.4f} out of bounds [0, 1]"
    
    print(f"  Grad-CAM Map dimensions: {overlay.shape}")
    print(f"  Grad-CAM Mean Intensity:  {res['cam_mean_intensity']:.4f}")
    print("  [OK] Grad-CAM visualization validated successfully.")

    print("\n========================================================")
    print("=== INTEGRATED RETINA MODULE VERIFICATION SUCCESSFUL ===")
    print("========================================================\n")

if __name__ == "__main__":
    verify_retina_module_acceptance()
