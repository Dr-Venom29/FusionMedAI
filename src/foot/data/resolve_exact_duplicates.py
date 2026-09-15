import os
import sys
import json
import hashlib
from pathlib import Path
from collections import defaultdict
import pandas as pd

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.foot.config import DATASET_ROOT, RAW_DATA, METADATA_DIR, METADATA_STATISTICS_DIR

LABEL_MAP = {
    "Grade 1": 0,
    "Grade 2": 1,
    "Grade 3": 2,
    "Grade 4": 3
}

def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def run_exact_duplicate_resolution():
    print("==================================================")
    print("Running Exact Duplicate Resolution")
    print("==================================================")
    
    # 1. Discover all image files
    all_files = list(RAW_DATA.rglob("*.jpg"))
    print(f"Total image files discovered in raw: {len(all_files):,}")
    assert len(all_files) == 10062, f"Expected 10,062 images, found {len(all_files)}"
    
    # 2. Extract metadata and compute SHA-256 hashes
    print("Computing SHA-256 hashes for all images...")
    records = []
    hash_to_files = defaultdict(list)
    
    for idx, filepath in enumerate(all_files, 1):
        rel_path = filepath.relative_to(RAW_DATA).as_posix()
        parts = rel_path.split('/')
        split_name = parts[0]
        grade_folder = parts[1]
        wagner_grade = LABEL_MAP[grade_folder]
        
        sha256_hash = compute_sha256(filepath)
        
        item = {
            "rel_path": rel_path,
            "split_name": split_name,
            "grade_folder": grade_folder,
            "wagner_grade": wagner_grade,
            "filename": filepath.name,
            "sha256": sha256_hash,
            "abs_path": str(filepath)
        }
        records.append(item)
        hash_to_files[sha256_hash].append(item)
        
        if idx % 2500 == 0 or idx == len(all_files):
            print(f"Processed {idx:,}/{len(all_files):,} images...")

    unique_hash_count = len(hash_to_files)
    print(f"\nUnique SHA-256 hashes: {unique_hash_count:,}")
    assert unique_hash_count == 10050, f"Expected 10,050 unique hashes, got {unique_hash_count}"
    
    # 3. Identify duplicate groups and select canonical files
    duplicate_groups = []
    canonical_manifest_rows = []
    cross_class_conflicts = 0
    
    # Define selection priority: train split > valid split > test split, then lexicographically lower path
    split_priority = {"train": 0, "valid": 1, "test": 2}
    
    group_counter = 0
    for sha256_hash, file_list in hash_to_files.items():
        if len(file_list) == 1:
            # Single unique file
            f = file_list[0]
            canonical_manifest_rows.append({
                "rel_path": f["rel_path"],
                "filename": f["filename"],
                "wagner_grade": f["wagner_grade"],
                "grade_folder": f["grade_folder"],
                "split_name": f["split_name"],
                "sha256": sha256_hash,
                "status": "KEPT",
                "role": "CANONICAL",
                "canonical_counterpart": f["rel_path"],
                "duplicate_group_id": None,
                "exclusion_reason": None
            })
        else:
            # Duplicate group (2 or more files)
            group_counter += 1
            group_id = f"EXACT_DUP_GROUP_{group_counter:03d}"
            
            # Check label consistency across duplicates in group
            grades = set(f["wagner_grade"] for f in file_list)
            if len(grades) > 1:
                cross_class_conflicts += 1
                print(f"WARNING: Cross-class duplicate found in {group_id}! Grades: {grades}")
                
            # Sort files by selection priority
            sorted_files = sorted(
                file_list,
                key=lambda x: (split_priority.get(x["split_name"], 99), x["rel_path"])
            )
            
            canonical_file = sorted_files[0]
            
            group_info = {
                "group_id": group_id,
                "sha256": sha256_hash,
                "total_copies": len(file_list),
                "wagner_grades": [int(g) for g in grades],
                "canonical_file": canonical_file["rel_path"],
                "excluded_files": [f["rel_path"] for f in sorted_files[1:]]
            }
            duplicate_groups.append(group_info)
            
            # Record canonical file entry
            canonical_manifest_rows.append({
                "rel_path": canonical_file["rel_path"],
                "filename": canonical_file["filename"],
                "wagner_grade": canonical_file["wagner_grade"],
                "grade_folder": canonical_file["grade_folder"],
                "split_name": canonical_file["split_name"],
                "sha256": sha256_hash,
                "status": "KEPT",
                "role": "CANONICAL",
                "canonical_counterpart": canonical_file["rel_path"],
                "duplicate_group_id": group_id,
                "exclusion_reason": None
            })
            
            # Record excluded duplicates entries
            for exc_file in sorted_files[1:]:
                canonical_manifest_rows.append({
                    "rel_path": exc_file["rel_path"],
                    "filename": exc_file["filename"],
                    "wagner_grade": exc_file["wagner_grade"],
                    "grade_folder": exc_file["grade_folder"],
                    "split_name": exc_file["split_name"],
                    "sha256": sha256_hash,
                    "status": "EXCLUDED",
                    "role": "EXCLUDED_EXACT_DUPLICATE",
                    "canonical_counterpart": canonical_file["rel_path"],
                    "duplicate_group_id": group_id,
                    "exclusion_reason": f"Exact SHA-256 duplicate of {canonical_file['rel_path']}"
                })

    df_manifest = pd.DataFrame(canonical_manifest_rows)
    
    kept_count = int((df_manifest["status"] == "KEPT").sum())
    excluded_count = int((df_manifest["status"] == "EXCLUDED").sum())
    
    print("\n--- Resolution Summary ---")
    print(f"Total Images Processed:       {len(df_manifest):,}")
    print(f"Canonical Unique Images Kept: {kept_count:,}")
    print(f"Exact Duplicates Excluded:    {excluded_count:,}")
    print(f"Exact Duplicate Groups:       {len(duplicate_groups):,}")
    print(f"Cross-Class Label Conflicts:  {cross_class_conflicts}")
    
    # 4. Acceptance Assertions
    assert kept_count == 10050, f"Expected 10,050 canonical images kept, got {kept_count}"
    assert excluded_count == 12, f"Expected 12 excluded duplicates, got {excluded_count}"
    assert len(duplicate_groups) == 12, f"Expected 12 duplicate groups, got {len(duplicate_groups)}"
    assert cross_class_conflicts == 0, f"Expected 0 cross-class conflicts, got {cross_class_conflicts}"
    
    # 5. Save Artifacts
    # JSON Report
    resolution_report = {
        "phase": "10.2.1",
        "title": "Exact Duplicate Resolution Report",
        "raw_dataset_status": "UNTOUCHED_IMMUTABLE",
        "total_files_scanned": len(all_files),
        "canonical_unique_images": int(kept_count),
        "exact_duplicates_excluded": int(excluded_count),
        "duplicate_groups_count": len(duplicate_groups),
        "cross_class_label_conflicts": cross_class_conflicts,
        "selection_rule": "Deterministic partition priority (train > valid > test), followed by lexicographically lowest relative path.",
        "duplicate_groups": duplicate_groups
    }
    
    json_path = METADATA_DIR / "duplicate_resolution.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(resolution_report, f, indent=2)
    print(f"\nSaved JSON report to: {json_path}")
    
    # CSV Table
    csv_path = METADATA_STATISTICS_DIR / "foot_duplicate_resolution.csv"
    df_manifest.to_csv(csv_path, index=False)
    print(f"Saved CSV manifest to: {csv_path}")
    
    print("\nAcceptance Checklist:")
    print(" [PASS] 10,050 canonical unique images selected")
    print(" [PASS] 12 duplicate groups documented")
    print(" [PASS] datasets/foot/raw/ untouched & unchanged")
    print(" [PASS] Every excluded duplicate mapped to canonical counterpart")
    print(" [PASS] 0 cross-class exact label conflicts verified")
    print("\n Exact Duplicate Resolution completed successfully!")

if __name__ == "__main__":
    run_exact_duplicate_resolution()
