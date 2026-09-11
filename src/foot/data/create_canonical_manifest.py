import os
import sys
import json
import re
from pathlib import Path
import pandas as pd

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.foot.config import DATASET_ROOT, RAW_DATA, PROCESSED_DATA, METADATA_DIR, METADATA_STATISTICS_DIR, CLASS_NAMES

def extract_source_image_id(filename: str) -> str:
    """
    Extracts underlying source image ID before Roboflow augmentation hash (.rf.<hash>).
    Example: '52109842_405285516886601_..._jpg.rf.0713f4891a686b6b622c1468aceb26cb.jpg'
             -> '52109842_405285516886601_..._jpg'
    """
    rf_match = re.split(r"\.rf\.[a-f0-9]{32}", filename, flags=re.IGNORECASE)
    if len(rf_match) > 1:
        return rf_match[0]
    return filename

def run_canonical_manifest_creation():
    print("==================================================")
    print("Running Phase 10.2.2 — Canonical Dataset Manifest Creation")
    print("==================================================")
    
    dup_res_csv = METADATA_STATISTICS_DIR / "foot_duplicate_resolution.csv"
    if not dup_res_csv.exists():
        raise FileNotFoundError(f"Prerequisite file not found: {dup_res_csv}. Please run Phase 10.2.1 first.")
        
    df_res = pd.read_csv(dup_res_csv)
    print(f"Loaded duplicate resolution records: {len(df_res):,} rows.")
    assert len(df_res) == 10062, f"Expected 10,062 rows in duplicate resolution, got {len(df_res)}"
    
    manifest_rows = []
    
    for idx, row in df_res.iterrows():
        rel_path = str(row["rel_path"])
        filename = str(row["filename"])
        wagner_grade = int(row["wagner_grade"])
        split_name = str(row["split_name"])
        sha256_hash = str(row["sha256"])
        status = str(row["status"])
        
        is_canonical = (status == "KEPT")
        source_id = extract_source_image_id(filename)
        class_name = CLASS_NAMES[wagner_grade]
        id_code = Path(filename).stem
        
        manifest_rows.append({
            "id_code": id_code,
            "image_path": rel_path,
            "source_image_id": source_id,
            "wagner_grade": wagner_grade,
            "class_name": class_name,
            "original_partition": split_name,
            "sha256": sha256_hash,
            "is_canonical": is_canonical
        })
        
    df_manifest = pd.DataFrame(manifest_rows)
    
    # 2. Validation & Summary Statistics
    total_records = len(df_manifest)
    canonical_records = df_manifest["is_canonical"].sum()
    excluded_records = (~df_manifest["is_canonical"]).sum()
    unique_source_ids = df_manifest["source_image_id"].nunique()
    canonical_source_ids = df_manifest[df_manifest["is_canonical"]]["source_image_id"].nunique()
    
    print("\n--- Canonical Manifest Summary ---")
    print(f"Total Database Rows:        {total_records:,}")
    print(f"Canonical Modeling Rows:   {canonical_records:,} (is_canonical == True)")
    print(f"Excluded Duplicate Rows:   {excluded_records:,} (is_canonical == False)")
    print(f"Unique Source Image Groups: {unique_source_ids:,} (Overall)")
    print(f"Canonical Source Groups:    {canonical_source_ids:,} (Canonical Subset)")
    
    # Assertions
    assert total_records == 10062, f"Expected 10,062 total records, got {total_records}"
    assert canonical_records == 10050, f"Expected 10,050 canonical records, got {canonical_records}"
    assert excluded_records == 12, f"Expected 12 excluded records, got {excluded_records}"
    assert unique_source_ids == 1770, f"Expected 1,770 source image groups, got {unique_source_ids}"
    
    # Check nulls
    assert df_manifest.isnull().sum().sum() == 0, "Manifest contains null values!"
    
    # 3. Save Artifacts
    PROCESSED_DATA.mkdir(parents=True, exist_ok=True)
    
    # Core modeling manifest
    processed_csv = PROCESSED_DATA / "canonical_manifest.csv"
    df_manifest.to_csv(processed_csv, index=False)
    print(f"\nSaved primary canonical manifest to: {processed_csv}")
    
    # Mirror in metadata statistics
    metadata_csv = METADATA_STATISTICS_DIR / "canonical_manifest.csv"
    df_manifest.to_csv(metadata_csv, index=False)
    print(f"Saved metadata mirror manifest to:   {metadata_csv}")
    
    # Summary JSON Report
    summary_report = {
        "phase": "10.2.2",
        "title": "Canonical Dataset Manifest Summary",
        "primary_manifest_path": str(processed_csv.relative_to(DATASET_ROOT)),
        "total_records": total_records,
        "canonical_modeling_records": int(canonical_records),
        "excluded_duplicate_records": int(excluded_records),
        "unique_source_image_groups": unique_source_ids,
        "class_breakdown_canonical": {
            class_name: int((df_manifest[df_manifest["is_canonical"]]["wagner_grade"] == grade_idx).sum())
            for grade_idx, class_name in enumerate(CLASS_NAMES)
        },
        "original_partition_breakdown_canonical": {
            part: int((df_manifest[df_manifest["is_canonical"]]["original_partition"] == part).sum())
            for part in ["train", "valid", "test"]
        },
        "columns": list(df_manifest.columns)
    }
    
    json_path = METADATA_DIR / "canonical_manifest_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)
    print(f"Saved JSON summary report to:        {json_path}")
    
    print("\nAcceptance Checklist:")
    print(" [PASS] 10,062 total records cataloged in authoritative manifest")
    print(" [PASS] 10,050 canonical modeling records identified (is_canonical == True)")
    print(" [PASS] 12 exact duplicates flagged (is_canonical == False)")
    print(" [PASS] 1,770 source_image_id groups extracted for group-stratified splitting")
    print(" [PASS] Manifest saved to datasets/foot/processed/canonical_manifest.csv")
    print("\nPhase 10.2.2 Canonical Dataset Manifest Creation completed successfully!")

if __name__ == "__main__":
    run_canonical_manifest_creation()
