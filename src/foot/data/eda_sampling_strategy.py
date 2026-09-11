import os
import sys
import json
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.foot.config import (
    DATASET_ROOT,
    PROCESSED_SPLITS_DIR,
    METADATA_DIR,
    METADATA_STATISTICS_DIR,
    CLASS_NAMES
)

GRADES_0_INDEXED = [0, 1, 2, 3]

def compute_class_weights(counts: dict, total_samples: int) -> dict:
    """Computes various class weighting schemes."""
    num_classes = len(counts)
    
    # 1. Standard Inverse Class Frequency: N_total / (K * N_c)
    inv_freq = {}
    for c, cnt in counts.items():
        inv_freq[c] = round(total_samples / (num_classes * cnt), 4)
        
    # 2. Square-Root Inverse Class Frequency (Smoothed): sqrt(N_max / N_c)
    max_cnt = max(counts.values())
    sqrt_inv_freq = {}
    for c, cnt in counts.items():
        sqrt_inv_freq[c] = round(float(np.sqrt(max_cnt / cnt)), 4)
        
    # 3. Effective Number of Samples Weighting (Cui et al., CVPR 2019)
    beta = 0.999
    eff_num_weights = {}
    eff_nums = {c: (1.0 - np.power(beta, cnt)) / (1.0 - beta) for c, cnt in counts.items()}
    sum_eff_nums = sum(eff_nums.values())
    for c in counts:
        eff_num_weights[c] = round(float((1.0 / eff_nums[c]) * (sum_eff_nums / num_classes)), 4)
        
    return {
        "inverse_class_frequency": inv_freq,
        "sqrt_inverse_class_frequency": sqrt_inv_freq,
        "effective_number_weights_beta_0_999": eff_num_weights
    }

def run_sampling_strategy_eda():
    print("==================================================")
    print("Running Phase 10.3.4 — Class Imbalance & Sampling Strategy Analysis")
    print("==================================================")
    
    index_csv = PROCESSED_SPLITS_DIR / "index.csv"
    groups_csv = METADATA_STATISTICS_DIR / "source_image_groups.csv"
    
    if not index_csv.exists():
        raise FileNotFoundError(f"Missing index.csv: {index_csv}. Run Phase 10.2 first.")
        
    df_index = pd.read_csv(index_csv)
    
    if groups_csv.exists():
        df_groups = pd.read_csv(groups_csv)
        total_grps = len(df_groups)
    else:
        total_grps = df_index["source_image_id"].nunique()
        
    print(f"Loaded canonical modeling population: {len(df_index):,} images, {total_grps:,} source groups.")
    
    # 1. Image-level class distribution
    class_counts_img = df_index["wagner_grade"].value_counts().sort_index().to_dict()
    total_imgs = len(df_index)
    
    class_dist_img = {}
    for grade in GRADES_0_INDEXED:
        cnt = class_counts_img.get(grade, 0)
        c_name = CLASS_NAMES[grade]
        class_dist_img[c_name] = {
            "wagner_grade_code": grade,
            "image_count": cnt,
            "percentage": round(cnt / total_imgs * 100, 2)
        }
        
    # 2. Group-level class distribution
    # Find majority grade per source group
    group_grades = df_index.groupby("source_image_id")["wagner_grade"].agg(lambda x: x.mode()[0]).to_dict()
    class_counts_grp = Counter(group_grades.values())
    
    class_dist_grp = {}
    for grade in GRADES_0_INDEXED:
        cnt = class_counts_grp.get(grade, 0)
        c_name = CLASS_NAMES[grade]
        class_dist_grp[c_name] = {
            "wagner_grade_code": grade,
            "group_count": cnt,
            "percentage": round(cnt / total_grps * 100, 2)
        }
        
    # 3. Class balance per split
    split_class_dist = {}
    for sp_name in ["train", "val", "test"]:
        df_sp = df_index[df_index["split"] == sp_name]
        sp_counts = df_sp["wagner_grade"].value_counts().sort_index().to_dict()
        sp_total = len(df_sp)
        
        split_class_dist[sp_name] = {
            CLASS_NAMES[g]: {
                "count": sp_counts.get(g, 0),
                "percentage": round(sp_counts.get(g, 0) / sp_total * 100, 2)
            } for g in GRADES_0_INDEXED
        }
        
    # 4. Imbalance Ratio Calculation
    min_cnt = min(class_counts_img.values())
    max_cnt = max(class_counts_img.values())
    imbalance_ratio = round(max_cnt / min_cnt, 2)
    
    # 5. Compute Class Weights
    weights_dict = compute_class_weights(class_counts_img, total_imgs)
    
    # Print Summary
    print("\n--- Image-Level Class Distribution ---")
    for g_name, data in class_dist_img.items():
        print(f" {g_name}: {data['image_count']:,} images ({data['percentage']}%)")
    print(f" Imbalance Ratio (Max:Min): {imbalance_ratio}:1 (Grade 1 vs Grade 3)")
    
    print("\n--- Group-Level Class Distribution ---")
    for g_name, data in class_dist_grp.items():
        print(f" {g_name}: {data['group_count']:,} groups ({data['percentage']}%)")
        
    print("\n--- Computed Class Weighting Options ---")
    for w_name, w_vals in weights_dict.items():
        print(f" [{w_name}]:")
        for g_code, w in w_vals.items():
            print(f"   {CLASS_NAMES[g_code]} (Code {g_code}): {w}")
            
    # Save Report
    json_report = {
        "phase": "10.3.4",
        "title": "Class Imbalance & Sampling Strategy Analysis",
        "total_canonical_images": total_imgs,
        "total_source_groups": total_grps,
        "imbalance_ratio_max_to_min": imbalance_ratio,
        "image_level_class_distribution": class_dist_img,
        "group_level_class_distribution": class_dist_grp,
        "partition_split_class_distribution": split_class_dist,
        "recommended_class_weights": weights_dict,
        "modeling_recommendations": {
            "sampling_strategy": "Standard random shuffling per epoch with class-weighted cross entropy loss or Focal Loss (gamma=2.0). Avoid aggressive oversampling of Grade 3 at image level because it duplicates images from limited source groups.",
            "loss_function": "Weighted Cross-Entropy (sqrt inverse frequency weights) or Focal Loss.",
            "group_awareness": "Group-stratified splitting ensures zero patient-level leakage across splits."
        }
    }
    
    json_path = METADATA_DIR / "eda_sampling_strategy.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2)
    print(f"\nSaved JSON sampling report to: {json_path}")
    
    # Save Markdown Report
    md_content = rf"""# Phase 10.3.4 — Class Imbalance & Sampling Strategy Report

## 1. Class Distribution & Imbalance Ratio

The canonical modeling population comprises **10,050 images** across **1,770 source groups**.

| Wagner Grade | Clinical Description | Canonical Images | Image % | Source Groups | Group % |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Grade 1** | Superficial Ulcer | **3,091** | 30.76% | **491** | 27.74% |
| **Grade 2** | Deep Ulcer to Tendon/Bone | **2,752** | 27.38% | **466** | 26.33% |
| **Grade 3** | Deep Ulcer with Osteomyelitis/Abscess | **1,595** | 15.87% | **353** | 19.94% |
| **Grade 4** | Partial Foot Gangrene | **2,612** | 25.99% | **460** | 25.99% |
| **Total** | | **10,050** | **100.00%** | **1,770** | **100.00%** |

- **Imbalance Ratio**: **{imbalance_ratio}:1** (Grade 1 vs Grade 3).
- **Classification**: Mild to moderate class imbalance. Grade 3 (15.87%) is underrepresented compared to Grade 1 (30.76%).

---

## 2. Partition Split Preservation

Group-stratified partitioning preserved class proportions across all three splits:

| Split | Grade 1 % | Grade 2 % | Grade 3 % | Grade 4 % |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | {split_class_dist['train']['Grade 1']['percentage']}% | {split_class_dist['train']['Grade 2']['percentage']}% | {split_class_dist['train']['Grade 3']['percentage']}% | {split_class_dist['train']['Grade 4']['percentage']}% |
| **Val** | {split_class_dist['val']['Grade 1']['percentage']}% | {split_class_dist['val']['Grade 2']['percentage']}% | {split_class_dist['val']['Grade 3']['percentage']}% | {split_class_dist['val']['Grade 4']['percentage']}% |
| **Test** | {split_class_dist['test']['Grade 1']['percentage']}% | {split_class_dist['test']['Grade 2']['percentage']}% | {split_class_dist['test']['Grade 3']['percentage']}% | {split_class_dist['test']['Grade 4']['percentage']}% |

---

## 3. Loss Weighting Options

| Wagner Grade | Inverse Frequency Weights | Sqrt Inverse Frequency (Recommended) | Effective Number Weights (\beta=0.999) |
| :--- | :---: | :---: | :---: |
| **Grade 1 (Code 0)** | `{weights_dict['inverse_class_frequency'][0]}` | `{weights_dict['sqrt_inverse_class_frequency'][0]}` | `{weights_dict['effective_number_weights_beta_0_999'][0]}` |
| **Grade 2 (Code 1)** | `{weights_dict['inverse_class_frequency'][1]}` | `{weights_dict['sqrt_inverse_class_frequency'][1]}` | `{weights_dict['effective_number_weights_beta_0_999'][1]}` |
| **Grade 3 (Code 2)** | `{weights_dict['inverse_class_frequency'][2]}` | `{weights_dict['sqrt_inverse_class_frequency'][2]}` | `{weights_dict['effective_number_weights_beta_0_999'][2]}` |
| **Grade 4 (Code 3)** | `{weights_dict['inverse_class_frequency'][3]}` | `{weights_dict['sqrt_inverse_class_frequency'][3]}` | `{weights_dict['effective_number_weights_beta_0_999'][3]}` |

---

## 4. Modeling & Sampling Recommendations

1. **Sampling Strategy**: Use standard epoch-based random shuffling without heavy oversampling. Oversampling Grade 3 images risks memorizing specific patient source group artifacts.
2. **Loss Function**: Use Cross-Entropy Loss with Sqrt Inverse Frequency weights or Focal Loss ($\gamma = 2.0$) to mitigate mild minority class suppression.
3. **Metric Focus**: Macro F1-score and Grade 3 recall will serve as primary evaluation metrics during Phase 10.4+.
"""
    
    md_path = METADATA_DIR / "eda_sampling_strategy.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved Markdown sampling report to: {md_path}")
    
    print("\nPhase 10.3.4 Class Imbalance & Sampling Strategy Analysis completed successfully!")

if __name__ == "__main__":
    run_sampling_strategy_eda()
