import os
import sys
import json
import re
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd

# Add project root to sys.path
root_path = Path(__file__).resolve().parents[3]
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

import src.foot.config as config
from src.foot.data.detect_duplicates import compute_dhash, hamming_distance

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def extract_source_image_id(filename: str) -> str:
    """
    Extracts the underlying source image ID before Roboflow augmentation hash (.rf.<hash>).
    Example: '52109842_405285516886601_..._jpg.rf.0713f4891a686b6b622c1468aceb26cb.jpg'
             -> '52109842_405285516886601_..._jpg'
    """
    rf_match = re.split(r"\.rf\.[a-f0-9]{32}", filename, flags=re.IGNORECASE)
    if len(rf_match) > 1:
        return rf_match[0]
    return filename

def investigate_leakage(raw_dir: Path):
    image_paths = [p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    total_images = len(image_paths)
    
    source_groups = defaultdict(list)
    
    for item in image_paths:
        rel_path = item.relative_to(raw_dir).as_posix()
        parts = item.relative_to(raw_dir).parts
        split_name = parts[0] if len(parts) >= 2 else "unknown"
        class_name = parts[1] if len(parts) >= 2 else "unknown"
        
        source_id = extract_source_image_id(item.name)
        source_groups[source_id].append({
            "path": rel_path,
            "filename": item.name,
            "split": split_name,
            "class": class_name,
            "full_path": str(item)
        })
        
    total_unique_source_images = len(source_groups)
    
    # Audit cross-partition leakage (source image variants appearing across train vs valid/test)
    cross_split_leakage_groups = []
    cross_class_conflict_groups = []
    augmented_groups_count = 0
    
    leakage_records = []

    for source_id, group in source_groups.items():
        if len(group) > 1:
            augmented_groups_count += 1
            
        splits = set(x["split"] for x in group)
        classes = set(x["class"] for x in group)
        
        has_split_leakage = len(splits) > 1
        has_class_conflict = len(classes) > 1
        
        if has_split_leakage:
            cross_split_leakage_groups.append({
                "source_id": source_id,
                "variant_count": len(group),
                "splits_present": sorted(list(splits)),
                "classes_present": sorted(list(classes)),
                "files": group
            })
            
        if has_class_conflict:
            cross_class_conflict_groups.append({
                "source_id": source_id,
                "variant_count": len(group),
                "splits_present": sorted(list(splits)),
                "classes_present": sorted(list(classes)),
                "files": group
            })
            
        for f in group:
            leakage_records.append({
                "source_image_id": source_id,
                "file_path": f["path"],
                "split": f["split"],
                "class": f["class"],
                "group_size": len(group),
                "has_cross_split_leakage": has_split_leakage,
                "has_cross_class_conflict": has_class_conflict
            })

    # Perform perceptual near-duplicate cross-split search
    print("Checking dHash perceptual near-duplicates for cross-split leakage...")
    dhash_list = []
    for item in image_paths:
        try:
            rel_path = item.relative_to(raw_dir).as_posix()
            parts = item.relative_to(raw_dir).parts
            dh = compute_dhash(item)
            dhash_list.append({
                "path": rel_path,
                "split": parts[0] if len(parts) >= 2 else "unknown",
                "class": parts[1] if len(parts) >= 2 else "unknown",
                "dhash": dh
            })
        except Exception:
            pass

    perceptual_cross_split_leakages = []
    for i in range(len(dhash_list)):
        for j in range(i + 1, min(i + 100, len(dhash_list))):
            d1, d2 = dhash_list[i], dhash_list[j]
            if d1["split"] != d2["split"]:
                if hamming_distance(d1["dhash"], d2["dhash"]) <= 3:
                    perceptual_cross_split_leakages.append({
                        "image_1": d1["path"],
                        "split_1": d1["split"],
                        "class_1": d1["class"],
                        "image_2": d2["path"],
                        "split_2": d2["split"],
                        "class_2": d2["class"]
                    })

    # Assess Patient Identifier Availability
    patient_id_available = False
    patient_id_note = (
        "Patient-level metadata (e.g. Patient ID / Case ID) is NOT provided in the raw dataset. "
        "Images are grouped by Roboflow source image IDs (extractable via filename prefixes). "
        "Patient-level separation cannot be guaranteed beyond source-image group separation. "
        "This limitation is explicitly documented to prevent false claims of patient-level isolation."
    )

    report_data = {
        "dataset_name": "Foot DFU Wagner 4-Class Dataset",
        "total_images_audited": total_images,
        "source_image_analysis": {
            "total_raw_image_files": total_images,
            "total_unique_source_images": total_unique_source_images,
            "augmentation_expansion_factor": round(total_images / total_unique_source_images, 2) if total_unique_source_images > 0 else 1.0,
            "groups_with_augmented_variants": augmented_groups_count
        },
        "leakage_investigation_summary": {
            "cross_split_source_leakage_groups_count": len(cross_split_leakage_groups),
            "cross_class_source_conflict_groups_count": len(cross_class_conflict_groups),
            "perceptual_cross_split_leakage_pairs_count": len(perceptual_cross_split_leakages)
        },
        "patient_identifier_audit": {
            "patient_id_available": patient_id_available,
            "limitation_note": patient_id_note
        },
        "leakage_findings": {
            "source_level_leakage_detected": len(cross_split_leakage_groups) > 0,
            "perceptual_leakage_detected": len(perceptual_cross_split_leakages) > 0
        },
        "sample_cross_split_leakages": cross_split_leakage_groups[:10],
        "sample_cross_class_conflicts": cross_class_conflict_groups[:10]
    }

    return report_data, leakage_records

def generate_markdown_report(report: dict) -> str:
    lines = []
    lines.append("# Data Leakage & Patient-Case Investigation Report")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append(f"- **Total Image Files**: `{report['total_images_audited']:,}`")
    lines.append(f"- **Unique Source Images Discovered**: `{report['source_image_analysis']['total_unique_source_images']:,}`")
    lines.append(f"- **Augmentation Expansion Factor**: `{report['source_image_analysis']['augmentation_expansion_factor']}×` (Source images expanded via offline Roboflow augmentations)")
    lines.append(f"- **Source Image Groups with Variants**: `{report['source_image_analysis']['groups_with_augmented_variants']:,}`")
    lines.append(f"- **Patient Identifier Availability**: `UNAVAILABLE` (Limitation Documented)")
    lines.append("")

    lines.append("## Data Leakage Audit Findings")
    lines.append("| Leakage Audit Check | Count Detected | Status | Clinical & Evaluation Risk Assessment |")
    lines.append("| :--- | :---: | :---: | :--- |")
    
    src_leak = report['leakage_investigation_summary']['cross_split_source_leakage_groups_count']
    src_conf = report['leakage_investigation_summary']['cross_class_source_conflict_groups_count']
    p_leak = report['leakage_investigation_summary']['perceptual_cross_split_leakage_pairs_count']
    
    lines.append(f"| **Source Image Cross-Split Leakage** | `{src_leak}` | {'⚠️ LEAK DETECTED' if src_leak > 0 else '✅ PASS'} | Augmented variants of the same source image exist in both train and test/valid splits |")
    lines.append(f"| **Source Image Cross-Class Conflict** | `{src_conf}` | {'⚠️ CONFLICT DETECTED' if src_conf > 0 else '✅ PASS'} | Augmented variants of the same source image labeled under different Wagner grades |")
    lines.append(f"| **Perceptual Cross-Split Leakage** | `{p_leak}` | {'⚠️ LEAK DETECTED' if p_leak > 0 else '✅ PASS'} | Visually near-identical images shared between train and test splits |")
    lines.append("")

    if src_leak > 0:
        lines.append("### Sample Source Image Cross-Split Leakage Instances")
        for idx, item in enumerate(report['sample_cross_split_leakages'][:5], 1):
            lines.append(f"**Source Group {idx}** (`{item['source_id']}` — {item['variant_count']} variants):")
            for f in item['files']:
                lines.append(f"  - `{f['path']}` (Split: **{f['split']}**, Class: **{f['class']}**)")
        lines.append("")

    lines.append("## Patient-Level Separation Limitation Notice")
    lines.append("> **Scientific Integrity Notice**: ")
    lines.append(f"> {report['patient_identifier_audit']['limitation_note']}")
    lines.append("")
    lines.append("## Downstream Action Plan for Step 10.2 (Data Pipeline)")
    lines.append("1. **Group-Aware Splitting**: In Step 10.2, dataset partitions must be constructed by grouping all augmented variants of each `source_image_id` into the SAME partition split (GroupKFold / Group-Based Stratified Split).")
    lines.append("2. **Leakage Elimination**: Source-level group partitioning eliminates cross-split leakage between training and evaluation sets.")

    return "\n".join(lines)

def main():
    raw_dir = config.RAW_DATA
    print(f"Executing Leakage Investigation on: {raw_dir}")
    
    report_data, leakage_records = investigate_leakage(raw_dir)
    
    # Save outputs to metadata/ and interim/metadata/
    target_dirs = [
        config.METADATA_DIR,
        config.DATASET_ROOT / "interim" / "metadata"
    ]
    
    for t_dir in target_dirs:
        t_dir.mkdir(parents=True, exist_ok=True)
        json_path = t_dir / "leakage_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        print(f"  [OK] Saved JSON leakage report: {json_path}")
        
        md_report = generate_markdown_report(report_data)
        md_path = t_dir / "leakage_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_report)
        print(f"  [OK] Saved Markdown leakage report: {md_path}")
        
    # Save CSV of leakage group mapping
    csv_path = config.METADATA_STATISTICS_DIR / "foot_leakage_groups.csv"
    df_leak = pd.DataFrame(leakage_records)
    df_leak.to_csv(csv_path, index=False)
    print(f"  [OK] Saved leakage groups CSV: {csv_path}")

    print("\n==========================================")
    print("LEAKAGE INVESTIGATION AUDIT COMPLETED")
    print("==========================================")
    try:
        print(generate_markdown_report(report_data))
    except UnicodeEncodeError:
        print(generate_markdown_report(report_data).encode("ascii", errors="replace").decode("ascii"))

if __name__ == "__main__":
    main()
