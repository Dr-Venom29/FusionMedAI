import sys
import json
from pathlib import Path
import pandas as pd

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import (
    METADATA_DIR,
    METADATA_QUALITY_DIR,
    METADATA_VALIDATION_DIR,
    METADATA_STATISTICS_DIR,
    TRAIN_CSV,
    TEST_CSV,
    TRAIN_IMAGES,
    TEST_IMAGES,
    NUM_CLASSES,
    CLASS_NAMES,
)

def main():
    # Load required initial datasets and pre-computed files
    if not TRAIN_CSV.exists():
        print(f"Error: train.csv not found at {TRAIN_CSV}")
        return 1
        
    df_train = pd.read_csv(TRAIN_CSV)
    expected_rows = len(df_train)
    
    # Load image sizes precomputed in Step 1.3
    image_sizes_path = METADATA_STATISTICS_DIR / "image_sizes.csv"
    if not image_sizes_path.exists():
        print(f"Error: image_sizes.csv not found at {image_sizes_path}. Run verify_dataset.py first.")
        return 1
        
    df_sizes = pd.read_csv(image_sizes_path)
    
    # Load verification report
    verification_report_path = METADATA_VALIDATION_DIR / "verification_report.json"
    if not verification_report_path.exists():
        print(f"Error: verification_report.json not found at {verification_report_path}. Run verify_dataset.py first.")
        return 1
        
    with open(verification_report_path, "r") as f:
        ver_report = json.load(f)
        
    # Merge train labels and image sizes
    df_train_with_fn = df_train.copy()
    df_train_with_fn["filename"] = df_train_with_fn["id_code"] + ".png"
    df_merged = pd.merge(df_train_with_fn, df_sizes, on="filename", how="inner")
    
    # Compute file sizes
    filesizes_bytes = []
    for filename in df_merged["filename"]:
        img_path = TRAIN_IMAGES / filename
        if img_path.exists():
            filesizes_bytes.append(img_path.stat().st_size)
        else:
            filesizes_bytes.append(0)
            
    df_merged["filesize_bytes"] = filesizes_bytes
    df_merged["filesize_kb"] = (df_merged["filesize_bytes"] / 1024).round().astype(int)
    df_merged["filesize_mb"] = (df_merged["filesize_bytes"] / (1024 * 1024)).round(4)
    df_merged["aspect_ratio"] = (df_merged["width"] / df_merged["height"]).round(2)
    
    # -------------------------------------------------------------
    # Step 1 — Generate train_metadata.csv
    # -------------------------------------------------------------
    df_train_metadata = pd.DataFrame({
        "id_code": df_merged["id_code"],
        "filename": df_merged["filename"],
        "label": df_merged["diagnosis"],
        "width": df_merged["width"],
        "height": df_merged["height"],
        "aspect_ratio": df_merged["aspect_ratio"],
        "filesize_kb": df_merged["filesize_kb"]
    })
    train_metadata_csv = METADATA_STATISTICS_DIR / "train_metadata.csv"
    df_train_metadata.to_csv(train_metadata_csv, index=False)
    
    # -------------------------------------------------------------
    # Step 2 — Generate class_distribution.csv
    # -------------------------------------------------------------
    severity_mapping = {i: name for i, name in enumerate(CLASS_NAMES)}
    dist = df_train_metadata["label"].value_counts().sort_index().reset_index()
    dist.columns = ["Label", "Count"]
    dist["Severity"] = dist["Label"].map(severity_mapping)
    # Reorder columns to Label, Severity, Count
    dist = dist[["Label", "Severity", "Count"]]
    class_dist_csv = METADATA_STATISTICS_DIR / "class_distribution.csv"
    dist.to_csv(class_dist_csv, index=False)
    
    # -------------------------------------------------------------
    # Step 3 — Generate dataset_statistics.json
    # -------------------------------------------------------------
    stats_widths = df_train_metadata["width"]
    stats_heights = df_train_metadata["height"]
    avg_filesize_kb = float(df_train_metadata["filesize_kb"].mean())
    min_filesize_kb = int(df_train_metadata["filesize_kb"].min())
    max_filesize_kb = int(df_train_metadata["filesize_kb"].max())
    avg_aspect_ratio = float(df_train_metadata["aspect_ratio"].mean())
    
    dataset_stats = {
        "dataset": "APTOS2019",
        "training_images": int(ver_report.get("train_images", 3662)),
        "testing_images": int(ver_report.get("test_images", 1928)),
        "num_classes": NUM_CLASSES,
        "labels": list(range(NUM_CLASSES)),
        "verification_passed": bool(ver_report.get("verification_passed", True)),
        "image_format": "PNG",
        "average_width": int(round(stats_widths.mean())),
        "average_height": int(round(stats_heights.mean())),
        "min_width": int(stats_widths.min()),
        "max_width": int(stats_widths.max()),
        "min_height": int(stats_heights.min()),
        "max_height": int(stats_heights.max()),
        "average_filesize_kb": round(avg_filesize_kb, 2),
        "minimum_filesize_kb": min_filesize_kb,
        "maximum_filesize_kb": max_filesize_kb,
        "average_aspect_ratio": round(avg_aspect_ratio, 2)
    }
    
    dataset_stats_json = METADATA_STATISTICS_DIR / "dataset_statistics.json"
    with open(dataset_stats_json, "w") as f:
        json.dump(dataset_stats, f, indent=2)
        
    # -------------------------------------------------------------
    # Step 4 — Generate image_statistics.csv
    # -------------------------------------------------------------
    df_image_statistics = pd.DataFrame({
        "id_code": df_merged["id_code"],
        "filename": df_merged["filename"],
        "diagnosis": df_merged["diagnosis"],
        "width": df_merged["width"],
        "height": df_merged["height"],
        "channels": 3,
        "aspect_ratio": df_merged["aspect_ratio"],
        "filesize_kb": df_merged["filesize_kb"]
    })
    image_stats_csv = METADATA_STATISTICS_DIR / "image_statistics.csv"
    df_image_statistics.to_csv(image_stats_csv, index=False)
    
    # -------------------------------------------------------------
    # Step 5 — Generate quality_statistics.csv
    # -------------------------------------------------------------
    df_quality_stats = pd.DataFrame({
        "Metric": [
            "RGB Images", 
            "Corrupted Images", 
            "Missing Images",
            "Average Width",
            "Average Height",
            "Average Aspect Ratio",
            "Average File Size (KB)"
        ],
        "Value": [
            int(ver_report.get("rgb_images", 3662)),
            int(ver_report.get("corrupted_images", 0)),
            int(ver_report.get("missing_images", 0)),
            int(round(stats_widths.mean())),
            int(round(stats_heights.mean())),
            round(avg_aspect_ratio, 2),
            round(avg_filesize_kb, 2)
        ]
    })
    quality_stats_csv = METADATA_QUALITY_DIR / "quality_statistics.csv"
    df_quality_stats.to_csv(quality_stats_csv, index=False)
    
    # -------------------------------------------------------------
    # Step 6 — Verify Metadata
    # -------------------------------------------------------------
    verification_success = True
    
    if len(df_train_metadata) != expected_rows:
        print(f"Verification warning: Expected {expected_rows} rows in train_metadata.csv, found {len(df_train_metadata)}")
        verification_success = False
        
    if len(dist) != NUM_CLASSES:
        print(f"Verification warning: Expected {NUM_CLASSES} rows in class_distribution.csv, found {len(dist)}")
        verification_success = False
        
    if not dataset_stats_json.exists():
        print("Verification warning: dataset_statistics.json is missing")
        verification_success = False
        
    if len(df_image_statistics) != expected_rows:
        print(f"Verification warning: Expected {expected_rows} rows in image_statistics.csv, found {len(df_image_statistics)}")
        verification_success = False
        
    # -------------------------------------------------------------
    # Step 7 — Console Report
    # -------------------------------------------------------------
    print("======================================")
    print("METADATA GENERATION")
    print("======================================")
    print()
    print(f"Training Images : {dataset_stats['training_images']}")
    print()
    print(f"Testing Images : {dataset_stats['testing_images']}")
    print()
    print(f"Classes : {dataset_stats['num_classes']}")
    print()
    print(f"Metadata Rows : {len(df_train_metadata)}")
    print()
    print("Image Statistics : Generated")
    print()
    print("Dataset Statistics : Generated")
    print()
    print("Class Distribution : Generated")
    print()
    print("Quality Statistics : Generated")
    print()
    if verification_success:
        print("Metadata Generation Completed")
    else:
        print("Metadata Generation Completed with Warnings")
        
    return 0

if __name__ == "__main__":
    main()
