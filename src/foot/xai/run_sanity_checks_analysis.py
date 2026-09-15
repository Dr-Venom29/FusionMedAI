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
from src.foot.xai.sanity_checks import (
    compute_cam_concentration,
    run_model_randomization_test,
    run_target_class_test
)

def run_sanity_checks_analysis():
    print("=============================================================")
    print("Phase 10.6.11 & 10.6.12 — Sanity Checks & Concentration Analysis")
    print("=============================================================\n", flush=True)
    
    exp_dir = Path(__file__).resolve().parents[3] / "experiments" / "foot" / "explainability"
    df_test = pd.read_csv(TEST_SPLIT_CSV)
    
    model = load_foot_final_model(device="cpu")
    gradcam = FootGradCAM(model=model, target_layer_name="backbone.features.8", device="cpu")
    transform = get_foot_test_transforms()
    
    # Reproducible stratified sampling across 4 Wagner grades with seed=42
    sample_dfs = []
    SEED = 42
    for g in range(4):
        g_df = df_test[df_test["wagner_grade"] == g]
        sample_dfs.append(g_df.sample(n=min(10, len(g_df)), random_state=SEED))
    df_sample = pd.concat(sample_dfs).reset_index(drop=True)
    
    top10_conc_list = []
    top20_conc_list = []
    rand_pcc_list = []
    class_pcc_list = []
    
    for idx, row in df_sample.iterrows():
        img_id = row["id_code"]
        true_grade = int(row["wagner_grade"])
        rel_path = row["image_path"]
        
        full_path = Path(RAW_DATA) / rel_path
        if not full_path.exists():
            full_path = Path(__file__).resolve().parents[3] / "datasets" / "foot" / "raw" / rel_path
            
        pil_img = Image.open(full_path).convert("RGB")
        tensor_img = transform(pil_img).unsqueeze(0)
        
        with torch.no_grad():
            out = model(tensor_img)
            pred_c = int(out["logits"].argmax(dim=1).item())
            
        cam, _ = gradcam.generate_cam(tensor_img, class_idx=pred_c)
        
        # 1. Concentration
        conc = compute_cam_concentration(cam, top_k_percentages=(0.10, 0.20))
        top10_conc_list.append(conc["top_10pct_concentration"])
        top20_conc_list.append(conc["top_20pct_concentration"])
        
        # 2. Model Randomization Test
        rand_res = run_model_randomization_test(
            trained_model=model,
            input_tensor=tensor_img,
            target_class=pred_c,
            target_layer_name="backbone.features.8"
        )
        if rand_res["randomization_pcc"] is not None:
            rand_pcc_list.append(rand_res["randomization_pcc"])
            
        # 3. Target Class Contrast Test
        alt_c = (pred_c + 1) % 4
        class_res = run_target_class_test(
            model=model,
            input_tensor=tensor_img,
            pred_class=pred_c,
            alt_class=alt_c,
            target_layer_name="backbone.features.8"
        )
        if class_res["class_contrast_pcc"] is not None:
            class_pcc_list.append(class_res["class_contrast_pcc"])
            
    sanity_summary = {
        "sample_size": len(df_sample),
        "seed": SEED,
        "mean_top_10pct_attribution_mass": round(float(np.mean(top10_conc_list)), 4),
        "mean_top_20pct_attribution_mass": round(float(np.mean(top20_conc_list)), 4),
        "model_randomization_mean_pcc": round(float(np.mean(rand_pcc_list)), 4) if len(rand_pcc_list) > 0 else None,
        "target_class_contrast_mean_pcc": round(float(np.mean(class_pcc_list)), 4) if len(class_pcc_list) > 0 else None,
        "qualification_note": "A predefined low-correlation heuristic (PCC < 0.30) was used to assess sensitivity of attributions to model parameter randomization."
    }
    
    with open(exp_dir / "sanity_checks.json", "w") as f:
        json.dump(sanity_summary, f, indent=2)
        
    print("=============================================================")
    print("Sanity Checks & Quantitative Attribution Summary:")
    print(f"  Sample Size: {sanity_summary['sample_size']} (Seed={SEED})")
    print(f"  Mean Attribution Mass in Top 10% Pixels: {sanity_summary['mean_top_10pct_attribution_mass']*100:.1f}%")
    print(f"  Mean Attribution Mass in Top 20% Pixels: {sanity_summary['mean_top_20pct_attribution_mass']*100:.1f}%")
    print(f"  Model Randomization Test Mean PCC: {sanity_summary['model_randomization_mean_pcc']}")
    print(f"  Target-Class Contrast Test Mean PCC: {sanity_summary['target_class_contrast_mean_pcc']}")
    print("=============================================================", flush=True)

if __name__ == "__main__":
    run_sanity_checks_analysis()
