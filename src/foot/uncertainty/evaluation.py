import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt

def evaluate_classwise_uncertainty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes classwise uncertainty metrics across Wagner Grades (Step 11).
    
    Generates tabular breakdown:
        Grade | Samples | Accuracy | Mean Confidence | Mean Entropy | Mean Variance | Mean MI
    """
    class_map = {0: "G1", 1: "G2", 2: "G3", 3: "G4"}
    records = []
    
    for grade_idx in range(4):
        label_str = class_map[grade_idx]
        sub = df[df["true_label"] == grade_idx]
        if len(sub) == 0:
            continue
            
        acc = float(np.mean(sub["predicted_label"] == sub["true_label"]))
        mean_conf = float(np.mean(sub["predictive_confidence"]))
        mean_ent = float(np.mean(sub["predictive_entropy"]))
        mean_var = float(np.mean(sub["predictive_variance"]))
        mean_mi = float(np.mean(sub["mutual_information"]))
        
        records.append({
            "Grade": label_str,
            "Grade_Index": grade_idx,
            "Samples": len(sub),
            "Accuracy": round(acc, 4),
            "Mean_Confidence": round(mean_conf, 4),
            "Mean_Entropy": round(mean_ent, 4),
            "Mean_Variance": round(mean_var, 6),
            "Mean_MI": round(mean_mi, 4)
        })
        
    return pd.DataFrame(records)


def evaluate_boundary_uncertainty(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluates Grade 2 vs Grade 3 boundary prediction uncertainty.
    """
    g2_mask = (df["true_label"] == 1)
    g3_mask = (df["true_label"] == 2)
    
    g2_correct = g2_mask & (df["predicted_label"] == 1)
    g2_incorrect = g2_mask & (df["predicted_label"] != 1)
    g3_correct = g3_mask & (df["predicted_label"] == 2)
    g3_incorrect = g3_mask & (df["predicted_label"] != 2)
    g2_as_g3 = g2_mask & (df["predicted_label"] == 2)
    g3_as_g2 = g3_mask & (df["predicted_label"] == 1)
    
    def sub_stats(mask):
        sub = df[mask]
        if len(sub) == 0:
            return {"count": 0, "variance_mean": 0.0, "entropy_mean": 0.0, "mi_mean": 0.0}
        return {
            "count": len(sub),
            "variance_mean": round(float(sub["predictive_variance"].mean()), 6),
            "entropy_mean": round(float(sub["predictive_entropy"].mean()), 4),
            "mi_mean": round(float(sub["mutual_information"].mean()), 4)
        }
        
    return {
        "G2_correct": sub_stats(g2_correct),
        "G2_incorrect": sub_stats(g2_incorrect),
        "G3_correct": sub_stats(g3_correct),
        "G3_incorrect": sub_stats(g3_incorrect),
        "G2_misclassified_as_G3": sub_stats(g2_as_g3),
        "G3_misclassified_as_G2": sub_stats(g3_as_g2)
    }


def extract_high_uncertainty_samples(df: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
    """
    Extracts high-uncertainty sample candidates (Top K by Entropy, Variance, and MI) (Step 12).
    """
    top_entropy = df.sort_values(by="predictive_entropy", ascending=False).head(top_k).copy()
    top_entropy["selection_reason"] = "top_entropy"
    
    top_variance = df.sort_values(by="predictive_variance", ascending=False).head(top_k).copy()
    top_variance["selection_reason"] = "top_variance"
    
    top_mi = df.sort_values(by="mutual_information", ascending=False).head(top_k).copy()
    top_mi["selection_reason"] = "top_mi"
    
    combined = pd.concat([top_entropy, top_variance, top_mi], ignore_index=True)
    combined = combined.drop_duplicates(subset=["id_code"]).reset_index(drop=True)
    return combined


def generate_uncertainty_explainability_overlays(
    model,
    vector_scaler,
    test_loader,
    high_unc_df: pd.DataFrame,
    output_dir: Path
) -> List[str]:
    """
    Connects uncertainty with explainability (Step 13) by building deterministic Grad-CAM heatmaps
    for high-uncertainty test samples.
    """
    from src.foot.xai.gradcam import FootGradCAM
    from src.foot.data.dataset import FootDFUDataset
    from src.foot.data.transforms import get_foot_test_transforms
    from src.foot.config import TEST_SPLIT_CSV, RAW_DATA, CLASS_NAMES
    import cv2


    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Explicitly ensure model is in eval mode for deterministic Grad-CAM explainability
    model.eval()
    gradcam = FootGradCAM(model=model, target_layer_name="backbone.features.8", device="cpu")
    
    # Map id_code to row in high_unc_df
    target_ids = set(high_unc_df["id_code"].values)
    
    test_ds = FootDFUDataset(csv_file=TEST_SPLIT_CSV, image_dir=RAW_DATA, transform=get_foot_test_transforms(), return_metadata=True)


    generated_files = []

    for i in range(len(test_ds)):
        _, _, meta = test_ds[i]
        id_code = meta["id_code"]
        if id_code not in target_ids:
            continue
            
        row = high_unc_df[high_unc_df["id_code"] == id_code].iloc[0]
        
        # Load unnormalized image & tensor for GradCAM
        orig_img = cv2.imread(meta["abs_path"])
        if orig_img is None:
            continue
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        
        # Get dataset transform tensor
        img_tensor, label = test_ds[i][:2]
        input_b = img_tensor.unsqueeze(0)
        
        target_class = int(row["predicted_label"])
        cam_mask, _ = gradcam.generate_cam(input_b, class_idx=target_class)

        
        # Overlay heatmap
        h, w, _ = orig_img.shape
        cam_resized = cv2.resize(cam_mask, (w, h))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        overlay = np.uint8(0.6 * orig_img + 0.4 * heatmap)
        
        # Resolve class names
        true_lbl_idx = int(row["true_label"])

        true_cls_name = (
            CLASS_NAMES[true_lbl_idx]
            if 0 <= true_lbl_idx < len(CLASS_NAMES)
            else f"Grade_{true_lbl_idx + 1}"
        )

        pred_cls_name = (
            CLASS_NAMES[target_class]
            if 0 <= target_class < len(CLASS_NAMES)
            else f"Grade_{target_class + 1}"
        )


        # Plot figure
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(orig_img)
        axes[0].set_title(f"True: {true_cls_name}\nPred: {pred_cls_name}", fontsize=9)
        axes[0].axis("off")
        
        axes[1].imshow(overlay)
        axes[1].set_title(f"Uncertainty Grad-CAM\nVar: {row['predictive_variance']:.5f} | Ent: {row['predictive_entropy']:.3f}", fontsize=9)
        axes[1].axis("off")
        
        plt.tight_layout()
        save_path = output_dir / f"high_unc_{id_code}.png"
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        generated_files.append(str(save_path))
        
    return generated_files
