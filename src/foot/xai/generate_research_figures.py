import sys
import json
import pandas as pd
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.foot.config import TEST_SPLIT_CSV, RAW_DATA
from src.foot.data.transforms import get_foot_test_transforms
from src.foot.models.factory import load_foot_final_model, build_foot_final_model
from src.foot.xai.gradcam import FootGradCAM
from src.foot.xai.visualization import overlay_heatmap, WAGNER_CLASSES

def generate_all_research_figures():
    print("=============================================================")
    print("Phase 10.6.15 — Generating Research Figures for Volume 06")
    print("=============================================================\n", flush=True)
    
    vol_img_dir = Path(__file__).resolve().parents[3] / "research" / "foot" / "Volume_06_Explainability" / "images"
    exp_dir = Path(__file__).resolve().parents[3] / "experiments" / "foot" / "explainability"
    vol_img_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = exp_dir / "explainability_results.csv"
    assert csv_path.exists(), f"Results CSV missing at: {csv_path}"
    df_res = pd.read_csv(csv_path)
    df_test = pd.read_csv(TEST_SPLIT_CSV)
    
    model = load_foot_final_model(device="cpu")
    gradcam = FootGradCAM(model=model, target_layer_name="backbone.features.8", device="cpu")
    transform = get_foot_test_transforms()
    
    # -------------------------------------------------------------
    # Figure 1: Correct Predictions by Class (Grade 1 .. 4)
    # -------------------------------------------------------------
    print("1. Generating Figure 1: Correct Predictions by Class...", flush=True)
    fig, axes = plt.subplots(4, 3, figsize=(12, 14))
    
    for g in range(4):
        sample = df_res[(df_res["true_class"] == g) & (df_res["correct"])].iloc[0]
        rel_path = df_test[df_test["id_code"] == sample["image_id"]].iloc[0]["image_path"]
        full_path = Path(RAW_DATA) / rel_path
        if not full_path.exists():
            full_path = Path(__file__).resolve().parents[3] / "datasets" / "foot" / "raw" / rel_path
            
        pil_img = Image.open(full_path).convert("RGB")
        tensor_img = transform(pil_img).unsqueeze(0)
        cam, _ = gradcam.generate_cam(tensor_img, class_idx=g)
        img_np = np.array(pil_img.resize((224, 224))).astype(np.float32) / 255.0
        overlay, heatmap = overlay_heatmap(img_np, cam)
        
        axes[g, 0].imshow(img_np)
        axes[g, 0].set_ylabel(f"Grade {g+1}", fontsize=12, fontweight="bold")
        if g == 0: axes[g, 0].set_title("Original RGB Image", fontsize=11, fontweight="bold")
        axes[g, 0].set_xticks([]); axes[g, 0].set_yticks([])
        
        axes[g, 1].imshow(heatmap)
        if g == 0: axes[g, 1].set_title("Grad-CAM Heatmap", fontsize=11, fontweight="bold")
        axes[g, 1].set_xticks([]); axes[g, 1].set_yticks([])
        
        axes[g, 2].imshow(overlay)
        if g == 0: axes[g, 2].set_title("Grad-CAM Overlay", fontsize=11, fontweight="bold")
        axes[g, 2].set_xticks([]); axes[g, 2].set_yticks([])
        
    plt.tight_layout()
    fig1_path = vol_img_dir / "fig1_qualitative_correct.png"
    fig.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved Figure 1 to: {fig1_path}", flush=True)
    
    # -------------------------------------------------------------
    # Figure 2: Incorrect Predictions by Class
    # -------------------------------------------------------------
    print("\n2. Generating Figure 2: Incorrect Predictions by Class...", flush=True)
    fig, axes = plt.subplots(4, 3, figsize=(12, 14))
    
    for g in range(4):
        err_samples = df_res[(df_res["true_class"] == g) & (~df_res["correct"])]
        if len(err_samples) > 0:
            sample = err_samples.iloc[0]
            pred_c = int(sample["predicted_class"])
        else:
            sample = df_res[df_res["true_class"] == g].iloc[0]
            pred_c = int(sample["predicted_class"])
            
        rel_path = df_test[df_test["id_code"] == sample["image_id"]].iloc[0]["image_path"]
        full_path = Path(RAW_DATA) / rel_path
        if not full_path.exists():
            full_path = Path(__file__).resolve().parents[3] / "datasets" / "foot" / "raw" / rel_path
            
        pil_img = Image.open(full_path).convert("RGB")
        tensor_img = transform(pil_img).unsqueeze(0)
        cam, _ = gradcam.generate_cam(tensor_img, class_idx=pred_c)
        img_np = np.array(pil_img.resize((224, 224))).astype(np.float32) / 255.0
        overlay, heatmap = overlay_heatmap(img_np, cam)
        
        axes[g, 0].imshow(img_np)
        axes[g, 0].set_ylabel(f"True Grade {g+1}\nPred Grade {pred_c+1}", fontsize=11, fontweight="bold")
        if g == 0: axes[g, 0].set_title("Original RGB Image", fontsize=11, fontweight="bold")
        axes[g, 0].set_xticks([]); axes[g, 0].set_yticks([])
        
        axes[g, 1].imshow(heatmap)
        if g == 0: axes[g, 1].set_title("Grad-CAM Heatmap", fontsize=11, fontweight="bold")
        axes[g, 1].set_xticks([]); axes[g, 1].set_yticks([])
        
        axes[g, 2].imshow(overlay)
        if g == 0: axes[g, 2].set_title("Grad-CAM Overlay", fontsize=11, fontweight="bold")
        axes[g, 2].set_xticks([]); axes[g, 2].set_yticks([])
        
    plt.tight_layout()
    fig2_path = vol_img_dir / "fig2_qualitative_incorrect.png"
    fig.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved Figure 2 to: {fig2_path}", flush=True)
    
    # -------------------------------------------------------------
    # Figure 3: Grade 2 <-> Grade 3 Error Breakdown
    # -------------------------------------------------------------
    print("\n3. Generating Figure 3: Grade 2 <-> Grade 3 Misclassification Comparison...", flush=True)
    g2_to_g3_samples = df_res[(df_res["true_class"] == 1) & (df_res["predicted_class"] == 2)]
    g3_to_g2_samples = df_res[(df_res["true_class"] == 2) & (df_res["predicted_class"] == 1)]
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Case A: True Grade 2 -> Pred Grade 3
    if len(g2_to_g3_samples) > 0:
        s_a = g2_to_g3_samples.iloc[0]
        rel_path_a = df_test[df_test["id_code"] == s_a["image_id"]].iloc[0]["image_path"]
        full_path_a = Path(RAW_DATA) / rel_path_a
        if not full_path_a.exists(): full_path_a = Path(__file__).resolve().parents[3] / "datasets" / "foot" / "raw" / rel_path_a
        pil_a = Image.open(full_path_a).convert("RGB")
        tensor_a = transform(pil_a).unsqueeze(0)
        cam_a, _ = gradcam.generate_cam(tensor_a, class_idx=2)
        img_np_a = np.array(pil_a.resize((224, 224))).astype(np.float32) / 255.0
        overlay_a, heatmap_a = overlay_heatmap(img_np_a, cam_a)
        
        axes[0, 0].imshow(img_np_a)
        axes[0, 0].set_ylabel("True Grade 2\nPred Grade 3", fontsize=11, fontweight="bold")
        axes[0, 0].set_title("Original RGB", fontsize=11, fontweight="bold")
        axes[0, 0].set_xticks([]); axes[0, 0].set_yticks([])
        
        axes[0, 1].imshow(heatmap_a)
        axes[0, 1].set_title("CAM (Target = Grade 3)", fontsize=11, fontweight="bold")
        axes[0, 1].set_xticks([]); axes[0, 1].set_yticks([])
        
        axes[0, 2].imshow(overlay_a)
        axes[0, 2].set_title("Overlay (Predicted Grade 3 Attribution)", fontsize=11, fontweight="bold")
        axes[0, 2].set_xticks([]); axes[0, 2].set_yticks([])
        
    # Case B: True Grade 3 -> Pred Grade 2
    if len(g3_to_g2_samples) > 0:
        s_b = g3_to_g2_samples.iloc[0]
        rel_path_b = df_test[df_test["id_code"] == s_b["image_id"]].iloc[0]["image_path"]
        full_path_b = Path(RAW_DATA) / rel_path_b
        if not full_path_b.exists(): full_path_b = Path(__file__).resolve().parents[3] / "datasets" / "foot" / "raw" / rel_path_b
        pil_b = Image.open(full_path_b).convert("RGB")
        tensor_b = transform(pil_b).unsqueeze(0)
        cam_b, _ = gradcam.generate_cam(tensor_b, class_idx=1)
        img_np_b = np.array(pil_b.resize((224, 224))).astype(np.float32) / 255.0
        overlay_b, heatmap_b = overlay_heatmap(img_np_b, cam_b)
        
        axes[1, 0].imshow(img_np_b)
        axes[1, 0].set_ylabel("True Grade 3\nPred Grade 2", fontsize=11, fontweight="bold")
        axes[1, 0].set_xticks([]); axes[1, 0].set_yticks([])
        
        axes[1, 1].imshow(heatmap_b)
        axes[1, 1].set_title("CAM (Target = Grade 2)", fontsize=11, fontweight="bold")
        axes[1, 1].set_xticks([]); axes[1, 1].set_yticks([])
        
        axes[1, 2].imshow(overlay_b)
        axes[1, 2].set_title("Overlay (Predicted Grade 2 Attribution)", fontsize=11, fontweight="bold")
        axes[1, 2].set_xticks([]); axes[1, 2].set_yticks([])
        
    plt.tight_layout()
    fig3_path = vol_img_dir / "fig3_grade2_3_confusion.png"
    fig.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved Figure 3 to: {fig3_path}", flush=True)
    
    # -------------------------------------------------------------
    # Figure 4: Synthetic Input Perturbation Sensitivity
    # -------------------------------------------------------------
    print("\n4. Generating Figure 4: Synthetic Input Perturbation Sensitivity...", flush=True)
    fig, axes = plt.subplots(4, 3, figsize=(12, 14))
    
    base_sample = df_res[df_res["correct"]].iloc[0]
    rel_path_q = df_test[df_test["id_code"] == base_sample["image_id"]].iloc[0]["image_path"]
    full_path_q = Path(RAW_DATA) / rel_path_q
    if not full_path_q.exists(): full_path_q = Path(__file__).resolve().parents[3] / "datasets" / "foot" / "raw" / rel_path_q
    orig_pil = Image.open(full_path_q).convert("RGB")
    
    quality_variants = {
        "Synthetic Baseline (Original)": orig_pil,
        "Synthetic Blur Perturbation": orig_pil.filter(ImageFilter.GaussianBlur(radius=3)),
        "Synthetic Low Contrast": ImageEnhance.Contrast(orig_pil).enhance(0.4),
        "Synthetic High Exposure": ImageEnhance.Brightness(orig_pil).enhance(1.6)
    }
    
    for i, (q_label, q_pil) in enumerate(quality_variants.items()):
        q_tensor = transform(q_pil).unsqueeze(0)
        with torch.no_grad():
            out_q = model(q_tensor)
            p_q = int(out_q["logits"].argmax(dim=1).item())
        cam_q, _ = gradcam.generate_cam(q_tensor, class_idx=p_q)
        img_np_q = np.array(q_pil.resize((224, 224))).astype(np.float32) / 255.0
        overlay_q, heatmap_q = overlay_heatmap(img_np_q, cam_q)
        
        axes[i, 0].imshow(img_np_q)
        axes[i, 0].set_ylabel(q_label, fontsize=11, fontweight="bold")
        if i == 0: axes[i, 0].set_title("Input Image", fontsize=11, fontweight="bold")
        axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])
        
        axes[i, 1].imshow(heatmap_q)
        if i == 0: axes[i, 1].set_title("Grad-CAM Heatmap", fontsize=11, fontweight="bold")
        axes[i, 1].set_xticks([]); axes[i, 1].set_yticks([])
        
        axes[i, 2].imshow(overlay_q)
        if i == 0: axes[i, 2].set_title("Overlay", fontsize=11, fontweight="bold")
        axes[i, 2].set_xticks([]); axes[i, 2].set_yticks([])
        
    plt.figtext(0.5, 0.01, "Illustrative synthetic perturbations used to examine attribution sensitivity; these are not dataset-quality annotations.", ha="center", fontsize=10, style="italic")
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig4_path = vol_img_dir / "fig4_synthetic_quality_perturbations.png"
    fig.savefig(fig4_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved Figure 4 to: {fig4_path}", flush=True)
    
    # -------------------------------------------------------------
    # Figure 5: Sanity Checks (Randomized Model & Class Contrast)
    # -------------------------------------------------------------
    print("\n5. Generating Figure 5: Sanity-Check Visual Verification...", flush=True)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    s_sanity = df_res[df_res["correct"]].iloc[0]
    rel_path_s = df_test[df_test["id_code"] == s_sanity["image_id"]].iloc[0]["image_path"]
    full_path_s = Path(RAW_DATA) / rel_path_s
    if not full_path_s.exists(): full_path_s = Path(__file__).resolve().parents[3] / "datasets" / "foot" / "raw" / rel_path_s
    pil_s = Image.open(full_path_s).convert("RGB")
    tensor_s = transform(pil_s).unsqueeze(0)
    target_c = int(s_sanity["predicted_class"])
    
    # Trained Model CAM
    cam_trained, _ = gradcam.generate_cam(tensor_s, class_idx=target_c)
    img_np_s = np.array(pil_s.resize((224, 224))).astype(np.float32) / 255.0
    overlay_tr, heatmap_tr = overlay_heatmap(img_np_s, cam_trained)
    
    # Randomized Model CAM
    rand_model = build_foot_final_model(pretrained=False)
    gradcam_rand = FootGradCAM(model=rand_model, target_layer_name="backbone.features.8", device="cpu")
    cam_rand, _ = gradcam_rand.generate_cam(tensor_s, class_idx=target_c)
    gradcam_rand.remove_hooks()
    overlay_rd, heatmap_rd = overlay_heatmap(img_np_s, cam_rand)
    
    # Alternative Class CAM
    alt_c = (target_c + 1) % 4
    cam_alt, _ = gradcam.generate_cam(tensor_s, class_idx=alt_c)
    overlay_alt, heatmap_alt = overlay_heatmap(img_np_s, cam_alt)
    
    # Row 1: Model Randomization Test
    axes[0, 0].imshow(img_np_s)
    axes[0, 0].set_ylabel("Model Randomization", fontsize=11, fontweight="bold")
    axes[0, 0].set_title("Input RGB Image", fontsize=11, fontweight="bold")
    axes[0, 0].set_xticks([]); axes[0, 0].set_yticks([])
    
    axes[0, 1].imshow(overlay_tr)
    axes[0, 1].set_title(f"Trained B3 Model (Class {target_c+1})", fontsize=11, fontweight="bold")
    axes[0, 1].set_xticks([]); axes[0, 1].set_yticks([])
    
    axes[0, 2].imshow(overlay_rd)
    axes[0, 2].set_title("Randomized Model (Adebayo Test)", fontsize=11, fontweight="bold")
    axes[0, 2].set_xticks([]); axes[0, 2].set_yticks([])
    
    # Row 2: Target-Class Contrast Test
    axes[1, 0].imshow(img_np_s)
    axes[1, 0].set_ylabel("Class Contrast Test", fontsize=11, fontweight="bold")
    axes[1, 0].set_xticks([]); axes[1, 0].set_yticks([])
    
    axes[1, 1].imshow(overlay_tr)
    axes[1, 1].set_title(f"Target: Pred Class {target_c+1}", fontsize=11, fontweight="bold")
    axes[1, 1].set_xticks([]); axes[1, 1].set_yticks([])
    
    axes[1, 2].imshow(overlay_alt)
    axes[1, 2].set_title(f"Target: Non-Pred Class {alt_c+1}", fontsize=11, fontweight="bold")
    axes[1, 2].set_xticks([]); axes[1, 2].set_yticks([])
    
    plt.tight_layout()
    fig5_path = vol_img_dir / "fig5_sanity_checks.png"
    fig.savefig(fig5_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved Figure 5 to: {fig5_path}", flush=True)
    
    print("\n=============================================================")
    print("SUCCESS: All 5 Research Figures Generated Successfully!")
    print("=============================================================", flush=True)

if __name__ == "__main__":
    generate_all_research_figures()
