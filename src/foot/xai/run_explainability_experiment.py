import sys
import json
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
from src.foot.xai.gradcam import FootGradCAM
from src.foot.xai.visualization import render_explanation_panel

def run_explainability_experiment():
    print("=============================================================")
    print("Phase 10.6.7 & 10.6.8 — Foot Ulcer Full Test Set Explainability")
    print("=============================================================\n", flush=True)
    
    # Base output directory (Option A: qualitative panels + CSV + summary)
    exp_dir = Path(__file__).resolve().parents[3] / "experiments" / "foot" / "explainability"
    qual_dir = exp_dir / "qualitative"
    
    for d in [exp_dir, qual_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    for g in range(1, 5):
        (qual_dir / f"Grade_{g}" / "correct").mkdir(parents=True, exist_ok=True)
        (qual_dir / f"Grade_{g}" / "incorrect").mkdir(parents=True, exist_ok=True)
        
    # Save experiment configuration
    config = {
        "modality": "foot",
        "task": "4-class Wagner diabetic foot ulcer classification",
        "model": "efficientnet_b3",
        "checkpoint": "experiments/foot/architecture_benchmark/efficientnet_b3/checkpoints/best_model.pt",
        "target_layer": "backbone.features.8",
        "input_resolution": [224, 224],
        "cam_method": "Grad-CAM",
        "high_attribution_threshold": 0.5
    }
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
        
    # Load dataset & model
    df_test = pd.read_csv(TEST_SPLIT_CSV)
    print(f"Loaded test split with {len(df_test)} samples.", flush=True)
    
    print("Loading frozen EfficientNet-B3 model...", flush=True)
    model = load_foot_final_model(device="cpu")
    gradcam = FootGradCAM(model=model, target_layer_name="backbone.features.8", device="cpu")
    transform = get_foot_test_transforms()
    
    records = []
    
    # Track counts for qualitative panel saving (~10 correct, ~5 incorrect per grade)
    qual_counts = {g: {"correct": 0, "incorrect": 0} for g in range(4)}
    QUAL_CORRECT_LIMIT = 10
    QUAL_INCORRECT_LIMIT = 5
    
    for idx, row in df_test.iterrows():
        img_id = row["id_code"]
        source_id = row["source_image_id"]
        true_grade = int(row["wagner_grade"])
        rel_path = row["image_path"]
        
        full_path = Path(RAW_DATA) / rel_path
        if not full_path.exists():
            full_path = Path(__file__).resolve().parents[3] / "datasets" / "foot" / "raw" / rel_path
            
        pil_img = Image.open(full_path).convert("RGB")
        tensor_img = transform(pil_img).unsqueeze(0)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            out = model(tensor_img)
            logits = out["logits"]
            probs = torch.softmax(logits, dim=1)
            pred_class = int(probs.argmax(dim=1).item())
            confidence = float(probs[0, pred_class].item())
            
        is_correct = (pred_class == true_grade)
        
        # Generate Grad-CAM for predicted class
        cam, _ = gradcam.generate_cam(tensor_img, class_idx=pred_class, target_size=(224, 224))
        
        # High-attribution area fraction (pixels >= 0.5)
        cam_mean = float(np.mean(cam))
        cam_max = float(np.max(cam))
        high_attr_area_frac = float(np.mean(cam >= 0.5))
        
        records.append({
            "image_id": img_id,
            "source_image_id": source_id,
            "true_class": true_grade,
            "predicted_class": pred_class,
            "correct": is_correct,
            "confidence": round(confidence, 4),
            "cam_target_class": pred_class,
            "cam_mean": round(cam_mean, 4),
            "cam_max": round(cam_max, 4),
            "high_attribution_area_fraction": round(high_attr_area_frac, 4)
        })
        
        # Qualitative panel saving
        grade_key = true_grade
        wagner_num = grade_key + 1 # Grade 1..4
        status_key = "correct" if is_correct else "incorrect"
        limit = QUAL_CORRECT_LIMIT if is_correct else QUAL_INCORRECT_LIMIT
        
        if qual_counts[grade_key][status_key] < limit:
            qual_counts[grade_key][status_key] += 1
            save_dir = qual_dir / f"Grade_{wagner_num}" / status_key
            img_np = np.array(pil_img.resize((224, 224))).astype(np.float32) / 255.0
            
            fig = render_explanation_panel(
                img=img_np,
                cam=cam,
                pred_class=pred_class,
                true_class=true_grade,
                save_path=str(save_dir / f"{img_id}_panel.png")
            )
            import matplotlib.pyplot as plt
            plt.close(fig)
            
        if (idx + 1) % 100 == 0 or (idx + 1) == len(df_test):
            print(f"Processed {idx + 1}/{len(df_test)} test images...", flush=True)
            
    # Convert records to DataFrame & save CSV
    results_df = pd.DataFrame(records)
    csv_path = exp_dir / "explainability_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV results to: {csv_path}", flush=True)
    
    # Compute summary metrics
    accuracy = float(results_df["correct"].mean())
    mean_attr_area_frac = float(results_df["high_attribution_area_fraction"].mean())
    correct_attr_area_frac = float(results_df[results_df["correct"]]["high_attribution_area_fraction"].mean())
    incorrect_attr_area_frac = float(results_df[~results_df["correct"]]["high_attribution_area_fraction"].mean())
    
    # Grade 2 <-> Grade 3 confusion stats
    g2_to_g3 = results_df[(results_df["true_class"] == 1) & (results_df["predicted_class"] == 2)]
    g3_to_g2 = results_df[(results_df["true_class"] == 2) & (results_df["predicted_class"] == 1)]
    
    summary = {
        "total_test_images": len(results_df),
        "overall_accuracy": round(accuracy, 4),
        "mean_high_attribution_area_fraction": round(mean_attr_area_frac, 4),
        "correct_cases_mean_high_attribution_area": round(correct_attr_area_frac, 4),
        "incorrect_cases_mean_high_attribution_area": round(incorrect_attr_area_frac, 4),
        "g2_predicted_as_g3_count": len(g2_to_g3),
        "g2_predicted_as_g3_mean_high_attribution_area": round(float(g2_to_g3["high_attribution_area_fraction"].mean()), 4) if len(g2_to_g3) > 0 else 0.0,
        "g3_predicted_as_g2_count": len(g3_to_g2),
        "g3_predicted_as_g2_mean_high_attribution_area": round(float(g3_to_g2["high_attribution_area_fraction"].mean()), 4) if len(g3_to_g2) > 0 else 0.0,
        "qualitative_panels_saved": qual_counts
    }
    
    with open(exp_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print("\n=============================================================")
    print("Explainability Experiment Summary:")
    print(f"  Total Test Images: {summary['total_test_images']}")
    print(f"  Overall Accuracy: {summary['overall_accuracy']:.4f}")
    print(f"  Mean High Attribution Area Fraction (>=0.5): {summary['mean_high_attribution_area_fraction']:.4f}")
    print(f"  Correct Cases Mean High Attribution Area: {summary['correct_cases_mean_high_attribution_area']:.4f}")
    print(f"  Incorrect Cases Mean High Attribution Area: {summary['incorrect_cases_mean_high_attribution_area']:.4f}")
    print(f"  Grade 2 -> Grade 3 Confusion Count: {summary['g2_predicted_as_g3_count']}")
    print(f"  Grade 3 -> Grade 2 Confusion Count: {summary['g3_predicted_as_g2_count']}")
    print("=============================================================", flush=True)

if __name__ == "__main__":
    run_explainability_experiment()
