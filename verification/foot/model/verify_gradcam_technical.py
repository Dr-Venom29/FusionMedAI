import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


from src.foot.config import TEST_SPLIT_CSV, RAW_DATA
from src.foot.data.transforms import get_foot_test_transforms
from src.foot.models.factory import load_foot_final_model
from src.foot.xai import FootGradCAM, overlay_heatmap

def verify_gradcam_technical():
    print("=============================================================")
    print("Phase 10.6.6 — Grad-CAM Technical Verification (8 Test Images)")
    print("=============================================================\n")
    
    # Load test split
    df_test = pd.read_csv(TEST_SPLIT_CSV)
    
    # Sample 2 images per class
    sample_dfs = []
    for grade in range(4):
        grade_samples = df_test[df_test["wagner_grade"] == grade].head(2)
        sample_dfs.append(grade_samples)
    df_sample = pd.concat(sample_dfs).reset_index(drop=True)
    
    print(f"Sampled {len(df_sample)} images across 4 Wagner classes for technical verification.")
    
    # Load model & GradCAM
    print("Loading frozen EfficientNet-B3 model...")
    model = load_foot_final_model(device="cpu")
    gradcam = FootGradCAM(model=model, target_layer_name="backbone.features.8", device="cpu")
    transform = get_foot_test_transforms()
    
    passed_count = 0
    
    for idx, row in df_sample.iterrows():
        img_id = row["id_code"]
        true_grade = row["wagner_grade"]
        rel_path = row["image_path"]
        full_path = Path(RAW_DATA) / rel_path
        
        if not full_path.exists():
            # Try alternate path under datasets/foot/raw
            full_path = Path(__file__).resolve().parents[3] / "datasets" / "foot" / "raw" / rel_path
            
        print(f"\n--- Image {idx+1}/{len(df_sample)}: {img_id} (True Grade {true_grade}) ---")
        assert full_path.exists(), f"Image path does not exist: {full_path}"
        
        # Load & transform
        pil_img = Image.open(full_path).convert("RGB")
        tensor_img = transform(pil_img).unsqueeze(0) # [1, 3, 224, 224]
        
        # Direct inference check
        model.eval()
        with torch.no_grad():
            out = model(tensor_img)
            logits = out["logits"]
            probs = torch.softmax(logits, dim=1)
            pred_class = int(probs.argmax(dim=1).item())
            confidence = float(probs[0, pred_class].item())
            
        print(f"   Inference output: Pred Grade={pred_class}, Conf={confidence:.4f}")
        
        # Generate Grad-CAM
        cam, cam_target_class = gradcam.generate_cam(tensor_img, class_idx=pred_class, target_size=(224, 224))
        
        # Technical checks
        checks = {
            "target_layer_exists": gradcam.target_layer is not None,
            "activation_captured": gradcam.activations is not None,
            "gradient_captured": gradcam.gradients is not None,
            "cam_shape_correct": cam.shape == (224, 224),
            "values_finite": np.isfinite(cam).all(),
            "values_normalized": (0.0 <= cam.min()) and (cam.max() <= 1.0),
            "spatial_variation": float(cam.max() - cam.min()) > 1e-6,
            "class_match": pred_class == cam_target_class
        }
        
        # Check overlay dimensions
        img_np = np.array(pil_img.resize((224, 224))).astype(np.float32) / 255.0
        overlay, heatmap = overlay_heatmap(img_np, cam)
        checks["overlay_dims_match"] = overlay.shape == (224, 224, 3)
        
        all_passed = all(checks.values())
        if all_passed:
            passed_count += 1
            print(f"   [OK] All technical checks passed! (CAM max={cam.max():.4f}, min={cam.min():.4f})")
        else:
            failed = [k for k, v in checks.items() if not v]
            print(f"   [FAIL] Verification FAILED for checks: {failed}")

            
    print("\n=============================================================")
    print(f"Verification Summary: {passed_count}/{len(df_sample)} images passed all checks.")
    print("=============================================================")
    
    assert passed_count == len(df_sample), "Technical verification failed for some images."

if __name__ == "__main__":
    verify_gradcam_technical()
