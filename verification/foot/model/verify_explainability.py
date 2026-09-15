import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
import numpy as np
from src.foot.models.factory import build_foot_final_model
from src.foot.xai import (
    FootGradCAM,
    overlay_heatmap,
    evaluate_pointing_game,
    evaluate_deletion_insertion
)

def test_explainability_module():
    print("=== Foot Ulcer Explainability Contract Verification ===")
    
    # 1. Instantiate model and GradCAM
    print("\n1. Instantiating EfficientNet-B3 model and FootGradCAM...")
    model = build_foot_final_model(pretrained=False)
    gradcam = FootGradCAM(model=model, target_layer_name="backbone.features.8", device="cpu")
    print(f"   Target layer resolved: {type(gradcam.target_layer)}")
    
    # 2. Synthetic forward & CAM generation
    print("\n2. Testing Grad-CAM generation on synthetic tensor [1, 3, 224, 224]...")
    x = torch.randn(1, 3, 224, 224)
    cam, pred_class = gradcam.generate_cam(x, class_idx=0, target_size=(224, 224))
    
    print(f"   Output CAM shape: {cam.shape}")
    print(f"   Output CAM range: min={cam.min():.4f}, max={cam.max():.4f}")
    print(f"   Predicted/Target Class: {pred_class}")
    
    assert cam.shape == (224, 224), f"Expected shape (224, 224), got {cam.shape}"
    assert 0.0 <= cam.min() and cam.max() <= 1.0, f"CAM out of range [0, 1]: min={cam.min()}, max={cam.max()}"
    assert isinstance(cam, np.ndarray), f"Expected np.ndarray, got {type(cam)}"
    
    # 3. Heatmap Overlay Test
    print("\n3. Testing heatmap overlay generation...")
    dummy_img = np.random.rand(224, 224, 3).astype(np.float32)
    overlay, heatmap = overlay_heatmap(dummy_img, cam, alpha=0.5)
    
    assert overlay.shape == (224, 224, 3), f"Overlay shape mismatch: {overlay.shape}"
    assert heatmap.shape == (224, 224, 3), f"Heatmap shape mismatch: {heatmap.shape}"
    assert 0.0 <= overlay.min() and overlay.max() <= 1.0, "Overlay values out of [0, 1]"
    print("   Overlay generation PASS")
    
    # 4. Pointing Game Evaluation Test
    print("\n4. Testing Pointing Game evaluation...")
    bbox = [10, 10, 200, 200]
    hit = evaluate_pointing_game(cam, bbox)
    print(f"   Pointing Game Hit: {hit}")
    assert isinstance(hit, bool), "Pointing Game output must be boolean"
    
    # 5. Deletion / Insertion AUC Test
    print("\n5. Testing Deletion and Insertion AUC metric calculation...")
    metrics = evaluate_deletion_insertion(model, x, cam, target_class=0, steps=3, device="cpu")
    print(f"   Deletion AUC: {metrics['deletion_auc']:.4f}")
    print(f"   Insertion AUC: {metrics['insertion_auc']:.4f}")
    assert "deletion_auc" in metrics and "insertion_auc" in metrics
    
    # 6. Hook Cleanup Test
    print("\n6. Testing hook removal and cleanup...")
    gradcam.remove_hooks()
    assert len(gradcam.handlers) == 0, "Handlers list not empty after cleanup"
    print("   Hook cleanup PASS")
    
    print("\n=============================================================")
    print("SUCCESS: Foot Ulcer Explainability Contract Passed!")
    print("=============================================================")

if __name__ == "__main__":
    test_explainability_module()
