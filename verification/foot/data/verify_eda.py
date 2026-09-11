import os
import sys
import json
from pathlib import Path
import pandas as pd

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.foot.config import (
    METADATA_DIR,
    METADATA_QUALITY_DIR,
    METADATA_STATISTICS_DIR,
    PROCESSED_SPLITS_DIR
)

def verify_eda_outputs():
    print("==================================================")
    print("Verifying Phase 10.3 — Foot EDA & Quality Artifacts")
    print("==================================================")
    
    errors = []
    
    # 1. Verify 10.3.1 Statistical Profiling JSON & CSV
    stats_json = METADATA_DIR / "eda_statistical_profiling.json"
    stats_csv = METADATA_STATISTICS_DIR / "foot_eda_stats.csv"
    
    if not stats_json.exists():
        errors.append(f"Missing statistical profiling JSON: {stats_json}")
    else:
        with open(stats_json, "r", encoding="utf-8") as f:
            data = json.load(f)
            if data.get("total_canonical_images") != 10050:
                errors.append(f"Expected 10,050 canonical images in {stats_json}, found {data.get('total_canonical_images')}")
        print(" [PASS] 10.3.1 Statistical Profiling JSON verified.")
        
    if not stats_csv.exists():
        errors.append(f"Missing statistical profiling CSV: {stats_csv}")
    else:
        df_stats = pd.read_csv(stats_csv)
        if len(df_stats) != 10050:
            errors.append(f"Expected 10,050 rows in {stats_csv}, found {len(df_stats)}")
        print(" [PASS] 10.3.1 Statistical Profiling CSV verified.")

    # 2. Verify 10.3.2 Class-Wise Visual Analysis Grid & JSON
    visual_grid = METADATA_QUALITY_DIR / "wagner_class_comparison_grid.png"
    visual_json = METADATA_DIR / "class_visual_analysis.json"
    
    if not visual_grid.exists():
        errors.append(f"Missing Wagner visual contact grid image: {visual_grid}")
    else:
        print(" [PASS] 10.3.2 Wagner Class Comparison Grid PNG verified.")
        
    if not visual_json.exists():
        errors.append(f"Missing class visual analysis JSON: {visual_json}")
    else:
        with open(visual_json, "r", encoding="utf-8") as f:
            vdata = json.load(f)
            if "class_profiles" not in vdata:
                errors.append(f"Missing class_profiles in {visual_json}")
        print(" [PASS] 10.3.2 Class Visual Analysis JSON verified.")

    # 3. Verify 10.3.3 Image Quality Analysis JSON & CSVs
    quality_json = METADATA_DIR / "eda_image_quality.json"
    quality_csv = METADATA_QUALITY_DIR / "quality_metrics_canonical.csv"
    outliers_csv = METADATA_QUALITY_DIR / "quality_outliers.csv"
    
    if not quality_json.exists():
        errors.append(f"Missing image quality JSON: {quality_json}")
    else:
        with open(quality_json, "r", encoding="utf-8") as f:
            qdata = json.load(f)
            if qdata.get("total_canonical_images") != 10050:
                errors.append(f"Expected 10,050 canonical images in {quality_json}")
        print(" [PASS] 10.3.3 Image Quality Analysis JSON verified.")
        
    if not quality_csv.exists():
        errors.append(f"Missing image quality CSV: {quality_csv}")
    else:
        df_q = pd.read_csv(quality_csv)
        if len(df_q) != 10050:
            errors.append(f"Expected 10,050 rows in {quality_csv}, found {len(df_q)}")
        print(" [PASS] 10.3.3 Canonical Quality Metrics CSV verified.")
        
    if not outliers_csv.exists():
        errors.append(f"Missing quality outliers CSV: {outliers_csv}")
    else:
        df_out = pd.read_csv(outliers_csv)
        print(f" [PASS] 10.3.3 Quality Outliers CSV verified ({len(df_out):,} outliers identified).")

    # 4. Verify 10.3.4 Class Imbalance & Sampling Strategy JSON & MD
    sampling_json = METADATA_DIR / "eda_sampling_strategy.json"
    sampling_md = METADATA_DIR / "eda_sampling_strategy.md"
    
    if not sampling_json.exists():
        errors.append(f"Missing sampling strategy JSON: {sampling_json}")
    else:
        with open(sampling_json, "r", encoding="utf-8") as f:
            sdata = json.load(f)
            if "imbalance_ratio_max_to_min" not in sdata:
                errors.append(f"Missing imbalance ratio in {sampling_json}")
        print(" [PASS] 10.3.4 Class Imbalance & Sampling Strategy JSON verified.")
        
    if not sampling_md.exists():
        errors.append(f"Missing sampling strategy Markdown report: {sampling_md}")
    else:
        print(" [PASS] 10.3.4 Sampling Strategy Markdown Report verified.")

    # 5. Verify 10.3.4 Outlier Analysis JSON & Manifest CSV
    outlier_analysis_json = METADATA_DIR / "eda_outlier_analysis.json"
    outlier_manifest_csv = METADATA_QUALITY_DIR / "outlier_analysis_manifest.csv"
    
    if not outlier_analysis_json.exists():
        errors.append(f"Missing outlier analysis JSON: {outlier_analysis_json}")
    else:
        with open(outlier_analysis_json, "r", encoding="utf-8") as f:
            odata = json.load(f)
            if "normal_distribution_count" not in odata:
                errors.append(f"Missing normal_distribution_count in {outlier_analysis_json}")
        print(" [PASS] 10.3.4 Outlier Analysis JSON verified.")
        
    if not outlier_manifest_csv.exists():
        errors.append(f"Missing outlier manifest CSV: {outlier_manifest_csv}")
    else:
        df_om = pd.read_csv(outlier_manifest_csv)
        if len(df_om) != 10050:
            errors.append(f"Expected 10,050 rows in {outlier_manifest_csv}, found {len(df_om)}")
        print(" [PASS] 10.3.4 Outlier Analysis Manifest CSV verified.")

    # 6. Verify 10.3.5 Class Separability Analysis JSON & Scatter Plot PNG
    separability_json = METADATA_DIR / "eda_class_separability.json"
    separability_png = METADATA_QUALITY_DIR / "class_separability_pca_tsne.png"
    
    if not separability_json.exists():
        errors.append(f"Missing class separability JSON: {separability_json}")
    else:
        with open(separability_json, "r", encoding="utf-8") as f:
            sepdata = json.load(f)
            if "pairwise_centroid_cosine_distances" not in sepdata:
                errors.append(f"Missing pairwise_centroid_cosine_distances in {separability_json}")
        print(" [PASS] 10.3.5 Class Separability Analysis JSON verified.")
        
    if not separability_png.exists():
        errors.append(f"Missing PCA/t-SNE scatter plot PNG: {separability_png}")
    else:
        print(" [PASS] 10.3.5 PCA & t-SNE Scatter Plot PNG verified.")

    # 7. Verify 10.3.6 Dataset Bias Analysis JSON & Heatmap PNG
    bias_json = METADATA_DIR / "eda_dataset_bias.json"
    bias_png = METADATA_QUALITY_DIR / "bias_correlation_matrix.png"
    
    if not bias_json.exists():
        errors.append(f"Missing dataset bias JSON: {bias_json}")
    else:
        with open(bias_json, "r", encoding="utf-8") as f:
            bdata = json.load(f)
            if "feature_correlation_audit" not in bdata:
                errors.append(f"Missing feature_correlation_audit in {bias_json}")
        print(" [PASS] 10.3.6 Dataset Bias Analysis JSON verified.")
        
    if not bias_png.exists():
        errors.append(f"Missing bias correlation matrix heatmap PNG: {bias_png}")
    else:
        print(" [PASS] 10.3.6 Bias Correlation Matrix Heatmap PNG verified.")

    # 8. Verify 10.3.7 Final EDA Decision JSON & MD in datasets/foot/metadata/eda/
    eda_decision_json = METADATA_DIR / "eda" / "reports" / "eda_final_decision.json"
    eda_decision_md = METADATA_DIR / "eda" / "reports" / "eda_final_decision.md"
    
    if not eda_decision_json.exists():
        errors.append(f"Missing final decision JSON: {eda_decision_json}")
    else:
        with open(eda_decision_json, "r", encoding="utf-8") as f:
            ddata = json.load(f)
            if ddata.get("acceptance_status") != "PASSED":
                errors.append(f"Expected acceptance_status PASSED in {eda_decision_json}")
        print(" [PASS] 10.3.7 Final EDA Decision JSON verified.")
        
    if not eda_decision_md.exists():
        errors.append(f"Missing final decision Markdown report: {eda_decision_md}")
    else:
        print(" [PASS] 10.3.7 Final EDA Decision Markdown Report verified.")

    if errors:
        print("\n[FAIL] Phase 10.3 Verification Errors:")
        for err in errors:
            print(f" - {err}")
        sys.exit(1)
    else:
        print("\n==================================================")
        print("ALL PHASE 10.3 EDA & DATASET QUALITY VERIFICATION CHECKS PASSED!")
        print("==================================================")

if __name__ == "__main__":
    verify_eda_outputs()
