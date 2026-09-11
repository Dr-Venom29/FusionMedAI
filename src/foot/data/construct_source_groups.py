import os
import sys
import json
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.foot.config import DATASET_ROOT, PROCESSED_DATA, METADATA_DIR, METADATA_STATISTICS_DIR, CLASS_NAMES

def run_source_group_construction():
    print("==================================================")
    print("Running Phase 10.2.3 — Source-Image Group Construction")
    print("==================================================")
    
    canonical_manifest_csv = PROCESSED_DATA / "canonical_manifest.csv"
    if not canonical_manifest_csv.exists():
        raise FileNotFoundError(f"Prerequisite file not found: {canonical_manifest_csv}. Please run Phase 10.2.2 first.")
        
    df_manifest = pd.read_csv(canonical_manifest_csv)
    print(f"Loaded canonical manifest: {len(df_manifest):,} total image records.")
    
    # Filter canonical modeling set (10,050 images)
    df_canonical = df_manifest[df_manifest["is_canonical"] == True].copy()
    print(f"Filtered canonical modeling set: {len(df_canonical):,} images.")
    
    # Group images by source_image_id
    groups = defaultdict(list)
    for idx, row in df_manifest.iterrows():
        source_id = str(row["source_image_id"])
        groups[source_id].append(row.to_dict())
        
    print(f"Total unique source-image groups constructed: {len(groups):,}")
    assert len(groups) == 1770, f"Expected 1,770 source groups, got {len(groups)}"
    
    # Analyze group sizes and grade consistency
    group_rows = []
    variant_counts = []
    canonical_variant_counts = []
    multi_class_groups = 0
    
    for group_idx, (source_id, items) in enumerate(groups.items(), 1):
        total_copies = len(items)
        canonical_items = [it for it in items if it["is_canonical"]]
        canonical_copies = len(canonical_items)
        
        variant_counts.append(total_copies)
        canonical_variant_counts.append(canonical_copies)
        
        # Wagner grades in group
        grades = [it["wagner_grade"] for it in items]
        grade_counts = Counter(grades)
        primary_grade = grade_counts.most_common(1)[0][0]
        primary_class = CLASS_NAMES[primary_grade]
        is_multi_class = (len(grade_counts) > 1)
        
        if is_multi_class:
            multi_class_groups += 1
            
        group_rows.append({
            "source_group_index": group_idx,
            "source_image_id": source_id,
            "primary_wagner_grade": primary_grade,
            "primary_class_name": primary_class,
            "total_raw_variants": total_copies,
            "canonical_modeling_variants": canonical_copies,
            "is_multi_class_group": is_multi_class,
            "grade_distribution": dict(grade_counts),
            "sample_image_path": canonical_items[0]["image_path"] if canonical_items else items[0]["image_path"]
        })
        
    df_groups = pd.DataFrame(group_rows)
    
    # Calculate statistics
    avg_raw_variants = sum(variant_counts) / len(variant_counts)
    avg_canonical_variants = sum(canonical_variant_counts) / len(canonical_variant_counts)
    min_variants = min(canonical_variant_counts)
    max_variants = max(canonical_variant_counts)
    
    print("\n--- Source Group Construction Summary ---")
    print(f"Total Unique Source-Image Groups: {len(df_groups):,}")
    print(f"Total Raw Images Grouped:          {sum(variant_counts):,}")
    print(f"Total Canonical Images Grouped:    {sum(canonical_variant_counts):,}")
    print(f"Mean Canonical Variants per Group: {avg_canonical_variants:.2f}")
    print(f"Variant Count Range (Canonical):   [{min_variants} .. {max_variants}]")
    print(f"Multi-Class Source Groups:         {multi_class_groups} (will use majority grade for group stratification)")
    
    # Assertions
    assert sum(variant_counts) == 10062, "Total raw grouped images != 10,062"
    assert sum(canonical_variant_counts) == 10050, "Total canonical grouped images != 10,050"
    assert len(df_groups) == 1770, "Total group count != 1,770"
    
    # Save Artifacts
    # CSV Table
    csv_path = METADATA_STATISTICS_DIR / "source_image_groups.csv"
    # Format grade distribution as string for CSV
    df_csv = df_groups.copy()
    df_csv["grade_distribution"] = df_csv["grade_distribution"].apply(lambda d: json.dumps(d))
    df_csv.to_csv(csv_path, index=False)
    print(f"\nSaved group statistics CSV to: {csv_path}")
    
    # JSON Summary Report
    summary_report = {
        "phase": "10.2.3",
        "title": "Source-Image Group Construction Report",
        "total_source_groups": len(df_groups),
        "total_raw_images": sum(variant_counts),
        "total_canonical_images": sum(canonical_variant_counts),
        "group_size_stats": {
            "mean_variants_per_group": round(avg_canonical_variants, 2),
            "min_variants": min_variants,
            "max_variants": max_variants
        },
        "multi_class_source_groups": multi_class_groups,
        "class_distribution_by_primary_group_label": {
            class_name: int((df_groups["primary_wagner_grade"] == g_idx).sum())
            for g_idx, class_name in enumerate(CLASS_NAMES)
        },
        "grouping_rule": "Extracted root source_image_id before Roboflow augmentation hash (.rf.<hash>). All augmented variants sharing the same source_image_id are bound strictly into the same group."
    }
    
    json_path = METADATA_DIR / "source_groups_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)
    print(f"Saved JSON group summary to:     {json_path}")
    
    print("\nAcceptance Checklist:")
    print(" [PASS] 1,770 unique source-image groups constructed")
    print(" [PASS] 10,050 canonical modeling images mapped to source groups")
    print(" [PASS] Group size distribution computed (avg ~5.68 variants/group)")
    print(" [PASS] Group primary label assigned for group-stratified partitioning")
    print(" [PASS] Source-image group structure verified & saved")
    print("\nPhase 10.2.3 Source-Image Group Construction completed successfully!")

if __name__ == "__main__":
    run_source_group_construction()
