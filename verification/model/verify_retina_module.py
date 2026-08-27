import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import src.config as config
from src.retina_module import RetinaModule

def verify_retina_module():
    """
    Verification script for the integrated RetinaModule wrapper, asserting
    that all components (inference, calibration, uncertainty, XAI) work
    together and produce the expected schema.
    """
    print("\n==================================================")
    print("Verification: Integrated Retina Module Inference")
    print("==================================================")
    
    # 1. Initialize RetinaModule
    checkpoint_path = Path(config.PROJECT_ROOT) / "experiments" / "efficientnet_b3" / "checkpoints" / "best_model.pt"
    print(f"Loading RetinaModule from checkpoint: {checkpoint_path.name}...")
    
    try:
        module = RetinaModule(
            checkpoint_path=checkpoint_path,
            device="cpu"  # Keep CPU-based for light verification
        )
        print("[OK] RetinaModule instantiated successfully.")
        print(f"  - Calibrated Temperature: {module.temperature:.4f}")
    except Exception as e:
        print(f"[FAIL] Failed to instantiate RetinaModule: {e}")
        sys.exit(1)
        
    # 2. Locate a test image
    test_images_dir = Path(config.PROJECT_ROOT) / "datasets" / "raw" / "aptos2019" / "train_images"
    sample_images = list(test_images_dir.glob("*.png"))
    
    if not sample_images:
        print(f"[FAIL] No test images found in {test_images_dir}. Download APTOS dataset first.")
        sys.exit(1)
        
    sample_image_path = sample_images[0]
    print(f"Found test image: {sample_image_path.name}")
    
    # 3. Execute integrated prediction
    mc_passes = 5 # use a small pass count for fast smoke verification
    print(f"Running prediction with {mc_passes} MC passes and Grad-CAM generation...")
    try:
        result = module.predict(
            image=sample_image_path,
            mc_passes=mc_passes,
            generate_cam=True
        )
        print("[OK] predict() call completed without errors.")
    except Exception as e:
        print(f"[FAIL] predict() call raised exception: {e}")
        sys.exit(1)
        
    # 4. Assert Output Schema
    print("Verifying output dictionary keys and types...")
    expected_keys = {
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
        "cam_mean_intensity": float
    }
    
    for key, expected_type in expected_keys.items():
        assert key in result, f"Missing key '{key}' in prediction results."
        assert isinstance(result[key], expected_type), (
            f"Type mismatch for '{key}': expected {expected_type}, got {type(result[key])}"
        )
    print("[OK] Schema keys and basic types verified.")
    
    # Check probabilities dimensions and properties
    assert len(result["raw_probabilities"]) == config.NUM_CLASSES, "Raw probabilities length mismatch."
    assert len(result["calib_probabilities"]) == config.NUM_CLASSES, "Calibrated probabilities length mismatch."
    assert np.allclose(sum(result["raw_probabilities"]), 1.0), "Raw probabilities do not sum to 1."
    assert np.allclose(sum(result["calib_probabilities"]), 1.0), "Calibrated probabilities do not sum to 1."
    print("[OK] Probability properties verified.")
    
    # Check uncertainty bounds
    assert 0 <= result["raw_confidence"] <= 1.0, f"raw_confidence out of bounds: {result['raw_confidence']}"
    assert 0 <= result["calib_confidence"] <= 1.0, f"calib_confidence out of bounds: {result['calib_confidence']}"
    assert 0 <= result["calib_entropy_norm"] <= 1.0, f"calib_entropy_norm out of bounds: {result['calib_entropy_norm']}"
    assert result["mc_predictive_entropy"] >= 0, f"mc_predictive_entropy is negative: {result['mc_predictive_entropy']}"
    assert result["mc_predictive_variance"] >= 0, f"mc_predictive_variance is negative: {result['mc_predictive_variance']}"
    # Allow small negative MI due to numerical precision
    assert result["mc_mutual_information"] >= -1e-6, f"mc_mutual_information out of bounds: {result['mc_mutual_information']}"
    print("[OK] Uncertainty bounds verified.")
    
    # Check XAI maps
    assert result["cam_overlay"] is not None, "cam_overlay is None."
    assert result["cam_heatmap"] is not None, "cam_heatmap is None."
    assert isinstance(result["cam_overlay"], np.ndarray), "cam_overlay is not a numpy array."
    assert isinstance(result["cam_heatmap"], np.ndarray), "cam_heatmap is not a numpy array."
    assert result["cam_overlay"].dtype == np.uint8, f"cam_overlay dtype is not uint8: {result['cam_overlay'].dtype}"
    assert result["cam_heatmap"].dtype == np.uint8, f"cam_heatmap dtype is not uint8: {result['cam_heatmap'].dtype}"
    assert 0.0 <= result["cam_mean_intensity"] <= 1.0, f"cam_mean_intensity out of bounds: {result['cam_mean_intensity']}"
    print("[OK] Grad-CAM visualization maps verified.")
    
    print("\n=== Prediction Result Summary ===")
    print(f"Image ID: {sample_image_path.name}")
    print(f"Predicted Diagnosis: Class {result['prediction']} ({result['prediction_label']})")
    print(f"Raw Confidence: {result['raw_confidence']:.2%}")
    print(f"Calibrated Confidence: {result['calib_confidence']:.2%}")
    print(f"Calibrated Normalized Entropy: {result['calib_entropy_norm']:.4f}")
    print(f"MC Predictive Entropy (Norm): {result['mc_predictive_entropy_norm']:.4f}")
    print(f"MC Predictive Variance: {result['mc_predictive_variance']:.6f}")
    print(f"MC Mutual Information: {result['mc_mutual_information']:.6f}")
    print(f"Grad-CAM Mean Intensity: {result['cam_mean_intensity']:.4f}")
    print(f"Total Inference Time: {result['latency_ms']:.2f} ms")
    print("=================================\n")
    
    print("=== INTEGRATED RETINA MODULE VERIFICATION SUCCESSFUL ===")
    print("========================================================\n")

if __name__ == "__main__":
    verify_retina_module()
