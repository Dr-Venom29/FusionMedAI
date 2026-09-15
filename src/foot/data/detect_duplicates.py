import os
import sys
import json
import hashlib
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image
import pandas as pd

# Add project root to sys.path
root_path = Path(__file__).resolve().parents[3]
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

import src.foot.config as config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def compute_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_dhash(image_path: Path, hash_size=8) -> int:
    """
    Computes a 64-bit Difference Hash (dHash) for an image.
    Resizes to (hash_size + 1, hash_size) grayscale and compares adjacent pixels.
    """
    with Image.open(image_path) as img:
        img = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
        pixels = list(img.getdata())
        
        difference = []
        for row in range(hash_size):
            for col in range(hash_size):
                pixel_left = pixels[row * (hash_size + 1) + col]
                pixel_right = pixels[row * (hash_size + 1) + col + 1]
                difference.append(pixel_left > pixel_right)
                
        decimal_value = 0
        for bit in difference:
            decimal_value = (decimal_value << 1) | int(bit)
        return decimal_value

def hamming_distance(hash1: int, hash2: int) -> int:
    return bin(hash1 ^ hash2).count("1")

def audit_duplicates(raw_dir: Path, near_duplicate_threshold=4):
    print("Collecting image paths for duplicate detection...")
    image_paths = [p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    total_images = len(image_paths)
    print(f"Found {total_images} images to audit.")
    
    sha256_map = defaultdict(list)
    dhash_map = []
    
    metadata_list = []
    
    for idx, item in enumerate(image_paths, 1):
        rel_path = item.relative_to(raw_dir).as_posix()
        parts = item.relative_to(raw_dir).parts
        split_name = parts[0] if len(parts) >= 2 else "unknown"
        class_name = parts[1] if len(parts) >= 2 else "unknown"
        
        # 1. Exact SHA-256 hash
        sha256_hash = compute_sha256(item)
        sha256_map[sha256_hash].append({
            "path": rel_path,
            "split": split_name,
            "class": class_name
        })
        
        # 2. Perceptual dHash
        try:
            dhash_val = compute_dhash(item)
            dhash_map.append({
                "path": rel_path,
                "split": split_name,
                "class": class_name,
                "dhash": dhash_val
            })
        except Exception as e:
            print(f"Warning: Failed dHash for {rel_path}: {e}")
            
    # Process Exact Duplicates
    exact_duplicate_groups = [items for h, items in sha256_map.items() if len(items) > 1]
    exact_duplicate_files_count = sum(len(group) for group in exact_duplicate_groups)
    
    exact_cross_split_leakage = []
    exact_cross_class_conflicts = []
    exact_duplicate_pairs = []
    
    for group in exact_duplicate_groups:
        splits = set(x["split"] for x in group)
        classes = set(x["class"] for x in group)
        
        if len(splits) > 1:
            exact_cross_split_leakage.append(group)
        if len(classes) > 1:
            exact_cross_class_conflicts.append(group)
            
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                exact_duplicate_pairs.append({
                    "image_1": group[i]["path"],
                    "split_1": group[i]["split"],
                    "class_1": group[i]["class"],
                    "image_2": group[j]["path"],
                    "split_2": group[j]["split"],
                    "class_2": group[j]["class"],
                    "match_type": "EXACT_SHA256",
                    "cross_split_leakage": group[i]["split"] != group[j]["split"],
                    "cross_class_conflict": group[i]["class"] != group[j]["class"]
                })

    # Process Near Duplicates using LSH buckets (16-bit prefix buckets for fast pairwise search)
    print("Searching for near-duplicates using LSH perceptual hashing...")
    lsh_buckets = defaultdict(list)
    for entry in dhash_map:
        # Mask top 16 bits for LSH bucket
        bucket_key = (entry["dhash"] >> 48) & 0xFFFF
        lsh_buckets[bucket_key].append(entry)
        
    near_duplicate_pairs = []
    near_cross_split_leakage = []
    near_cross_class_conflicts = []
    
    seen_near_pairs = set()
    
    for bucket_key, entries in lsh_buckets.items():
        if len(entries) < 2:
            continue
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                e1, e2 = entries[i], entries[j]
                if e1["path"] == e2["path"]:
                    continue
                pair_key = tuple(sorted([e1["path"], e2["path"]]))
                if pair_key in seen_near_pairs:
                    continue
                    
                dist = hamming_distance(e1["dhash"], e2["dhash"])
                if dist <= near_duplicate_threshold:
                    seen_near_pairs.add(pair_key)
                    is_cross_split = e1["split"] != e2["split"]
                    is_cross_class = e1["class"] != e2["class"]
                    
                    near_duplicate_pairs.append({
                        "image_1": e1["path"],
                        "split_1": e1["split"],
                        "class_1": e1["class"],
                        "image_2": e2["path"],
                        "split_2": e2["split"],
                        "class_2": e2["class"],
                        "dhash_hamming_distance": dist,
                        "match_type": "NEAR_DUPLICATE_DHASH",
                        "cross_split_leakage": is_cross_split,
                        "cross_class_conflict": is_cross_class
                    })
                    
                    if is_cross_split:
                        near_cross_split_leakage.append(pair_key)
                    if is_cross_class:
                        near_cross_class_conflicts.append(pair_key)

    unique_exact_images_count = len(sha256_map)

    report_data = {
        "dataset_name": "Foot DFU Wagner 4-Class Dataset",
        "total_images_audited": total_images,
        "exact_duplicate_summary": {
            "total_exact_duplicate_groups": len(exact_duplicate_groups),
            "total_exact_duplicate_files": exact_duplicate_files_count,
            "unique_exact_images": unique_exact_images_count,
            "exact_cross_split_leakages_count": len(exact_cross_split_leakage),
            "exact_cross_class_conflicts_count": len(exact_cross_class_conflicts)
        },
        "near_duplicate_summary": {
            "near_duplicate_pairs_count": len(near_duplicate_pairs),
            "near_cross_split_leakages_count": len(near_cross_split_leakage),
            "near_cross_class_conflicts_count": len(near_cross_class_conflicts),
            "dhash_hamming_threshold": near_duplicate_threshold
        },
        "critical_findings": {
            "data_leakage_risk": len(exact_cross_split_leakage) > 0 or len(near_cross_split_leakage) > 0,
            "label_conflict_risk": len(exact_cross_class_conflicts) > 0 or len(near_cross_class_conflicts) > 0
        },
        "exact_cross_split_details": exact_cross_split_leakage[:10],
        "exact_cross_class_details": exact_cross_class_conflicts[:10]
    }

    return report_data, exact_duplicate_pairs + near_duplicate_pairs

def generate_markdown_report(report: dict) -> str:
    lines = []
    lines.append("# Duplicate Detection & Leakage Audit Report")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append(f"- **Total Images Audited**: `{report['total_images_audited']:,}`")
    lines.append(f"- **Unique Exact Images**: `{report['exact_duplicate_summary']['unique_exact_images']:,}`")
    lines.append(f"- **Exact Duplicate Groups**: `{report['exact_duplicate_summary']['total_exact_duplicate_groups']}`")
    lines.append(f"- **Exact Duplicate Files Count**: `{report['exact_duplicate_summary']['total_exact_duplicate_files']}`")
    lines.append(f"- **Near Duplicate Pairs (dHash $\\le 4$)**: `{report['near_duplicate_summary']['near_duplicate_pairs_count']}`")
    lines.append("")

    lines.append("## Data Leakage & Label Conflict Audit")
    lines.append("| Audit Check | Count Detected | Status | Impact / Risk Assessment |")
    lines.append("| :--- | :---: | :---: | :--- |")
    
    exact_leak = report['exact_duplicate_summary']['exact_cross_split_leakages_count']
    exact_conf = report['exact_duplicate_summary']['exact_cross_class_conflicts_count']
    near_leak = report['near_duplicate_summary']['near_cross_split_leakages_count']
    near_conf = report['near_duplicate_summary']['near_cross_class_conflicts_count']
    
    lines.append(f"| **Exact Cross-Split Leakage** | `{exact_leak}` | {'⚠️ LEAK DETECTED' if exact_leak > 0 else '✅ PASS'} | Duplicate image exists in both train and test/valid splits |")
    lines.append(f"| **Exact Cross-Class Conflict** | `{exact_conf}` | {'⚠️ CONFLICT DETECTED' if exact_conf > 0 else '✅ PASS'} | Identical image assigned to multiple different Wagner grades |")
    lines.append(f"| **Near Cross-Split Leakage** | `{near_leak}` | {'⚠️ LEAK DETECTED' if near_leak > 0 else '✅ PASS'} | Visually identical image across train/test partitions |")
    lines.append(f"| **Near Cross-Class Conflict** | `{near_conf}` | {'⚠️ CONFLICT DETECTED' if near_conf > 0 else '✅ PASS'} | Visually identical image assigned to different Wagner grades |")
    lines.append("")

    if exact_leak > 0:
        lines.append("### Sample Cross-Split Data Leakage Instances")
        for idx, group in enumerate(report['exact_cross_split_details'][:5], 1):
            lines.append(f"**Group {idx}**:")
            for item in group:
                lines.append(f"  - `{item['path']}` (Split: **{item['split']}**, Class: **{item['class']}**)")
        lines.append("")

    if exact_conf > 0:
        lines.append("### Sample Cross-Class Label Conflict Instances")
        for idx, group in enumerate(report['exact_cross_class_details'][:5], 1):
            lines.append(f"**Group {idx}**:")
            for item in group:
                lines.append(f"  - `{item['path']}` (Split: **{item['split']}**, Class: **{item['class']}**)")
        lines.append("")

    lines.append("## Rule Compliance & Immutability Notice")
    lines.append("> **Immutability Enforced**: No files were removed or altered inside `datasets/foot/raw/`. Duplicate groups and leakage instances are recorded in `duplicate_report.json` and `foot_duplicate_pairs.csv` to inform downstream dataset split construction in Step 10.2.")

    return "\n".join(lines)

def main():
    raw_dir = config.RAW_DATA
    print(f"Executing Duplicate Detection on: {raw_dir}")
    
    report_data, duplicate_pairs = audit_duplicates(raw_dir)
    
    # Save outputs to metadata/ and interim/metadata/
    target_dirs = [
        config.METADATA_DIR,
        config.DATASET_ROOT / "interim" / "metadata"
    ]
    
    for t_dir in target_dirs:
        t_dir.mkdir(parents=True, exist_ok=True)
        json_path = t_dir / "duplicate_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        print(f"  [OK] Saved JSON duplicate report: {json_path}")
        
        md_report = generate_markdown_report(report_data)
        md_path = t_dir / "duplicate_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_report)
        print(f"  [OK] Saved Markdown duplicate report: {md_path}")
        
    # Save CSV of all duplicate pairs
    csv_path = config.METADATA_STATISTICS_DIR / "foot_duplicate_pairs.csv"
    df_dup = pd.DataFrame(duplicate_pairs)
    df_dup.to_csv(csv_path, index=False)
    print(f"  [OK] Saved duplicate pairs CSV: {csv_path}")

    print("\n==========================================")
    print("DUPLICATE DETECTION AUDIT COMPLETED")
    print("==========================================")
    try:
        print(generate_markdown_report(report_data))
    except UnicodeEncodeError:
        print(generate_markdown_report(report_data).encode("ascii", errors="replace").decode("ascii"))

if __name__ == "__main__":
    main()
