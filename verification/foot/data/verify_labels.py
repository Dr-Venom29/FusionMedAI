import os
import sys
import json
from pathlib import Path
from collections import Counter, defaultdict

# Add project root to sys.path
root_path = Path(__file__).resolve().parents[3]
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

import src.foot.config as config

EXPECTED_CLASSES = ["Grade 1", "Grade 2", "Grade 3", "Grade 4"]
LABEL_MAPPING = {
    "Grade 1": 0,
    "Grade 2": 1,
    "Grade 3": 2,
    "Grade 4": 3
}

def audit_labels(raw_dir: Path):
    issues = []
    warnings = []
    
    hidden_files = []
    unexpected_folders = []
    split_class_folders = defaultdict(set)
    class_image_counts = defaultdict(lambda: defaultdict(int))
    
    # Track all paths
    for item in raw_dir.rglob("*"):
        rel_path = item.relative_to(raw_dir).as_posix()
        
        # Check hidden files
        if item.name.startswith(".") or item.name.startswith("Thumbs.db"):
            hidden_files.append(rel_path)
            warnings.append(f"Hidden or OS system file found: {rel_path}")
            
        parts = item.relative_to(raw_dir).parts
        
        # Audit folder structure
        if item.is_dir():
            if len(parts) == 1:
                # Top level split folders (expected: train, valid, test)
                if parts[0].lower() not in ["train", "valid", "test", "val"]:
                    unexpected_folders.append(rel_path)
                    issues.append(f"Unexpected top-level folder in raw/: '{parts[0]}'")
            elif len(parts) == 2:
                split_name = parts[0]
                class_folder = parts[1]
                split_class_folders[split_name].add(class_folder)
                
                if class_folder not in EXPECTED_CLASSES:
                    unexpected_folders.append(rel_path)
                    issues.append(f"Unexpected class folder name in split '{split_name}': '{class_folder}'")
            elif len(parts) > 2:
                # Unexpected deep nested folder
                unexpected_folders.append(rel_path)
                issues.append(f"Unexpected deeply nested folder: '{rel_path}'")
        else:
            # File audit
            if len(parts) >= 3:
                split_name = parts[0]
                class_folder = parts[1]
                if item.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    class_image_counts[split_name][class_folder] += 1

    # Check consistency of class names across splits
    all_discovered_classes = set()
    for split, classes in split_class_folders.items():
        all_discovered_classes.update(classes)
        missing_in_split = set(EXPECTED_CLASSES) - classes
        if missing_in_split:
            issues.append(f"Split '{split}' is missing expected class folders: {missing_in_split}")
            
    # Check for duplicate class representations (case insensitivity or formatting variants)
    normalized_classes = [c.lower().replace(" ", "").replace("_", "") for c in all_discovered_classes]
    if len(normalized_classes) != len(set(normalized_classes)):
        issues.append(f"Duplicate/conflicting class representations detected among: {all_discovered_classes}")
        
    # Check for empty classes
    empty_classes = []
    for split, cls_counts in class_image_counts.items():
        for expected_cls in EXPECTED_CLASSES:
            count = cls_counts.get(expected_cls, 0)
            if count == 0:
                empty_classes.append(f"{split}/{expected_cls}")
                issues.append(f"Class '{expected_cls}' in split '{split}' contains 0 images.")

    # Audit metadata text files
    metadata_conflict = False
    readme_dataset = raw_dir / "README.dataset.txt"
    readme_roboflow = raw_dir / "README.roboflow.txt"
    
    metadata_notes = []
    if readme_dataset.exists():
        text = readme_dataset.read_text(encoding="utf-8", errors="ignore")
        metadata_notes.append(f"README.dataset.txt: {text.strip()}")
    if readme_roboflow.exists():
        text = readme_roboflow.read_text(encoding="utf-8", errors="ignore")
        if "Four groups" not in text and "10062" not in text:
            warnings.append("README.roboflow.txt text does not explicitly match expected image count or groups.")

    passed = len(issues) == 0

    audit_result = {
        "status": "PASS" if passed else "FAIL",
        "folder_to_label_mapping": LABEL_MAPPING,
        "expected_classes": EXPECTED_CLASSES,
        "discovered_classes": sorted(list(all_discovered_classes)),
        "class_counts_by_split": {k: dict(v) for k, v in class_image_counts.items()},
        "issues_found": issues,
        "warnings_found": warnings,
        "hidden_files": hidden_files,
        "unexpected_folders": unexpected_folders,
        "empty_classes": empty_classes,
        "duplicate_class_representations": len(all_discovered_classes) != len(set(normalized_classes)),
        "metadata_conflict": metadata_conflict,
        "metadata_summary": metadata_notes
    }
    
    return audit_result

def generate_markdown_report(audit: dict) -> str:
    lines = []
    lines.append("# Label Verification Report")
    lines.append("")
    lines.append(f"**Overall Status**: `{audit['status']}`")
    lines.append("")
    lines.append("## Explicit Folder-to-Label Mapping")
    lines.append("| Folder Name | Class Index | Clinical Description |")
    lines.append("| :--- | :--- | :--- |")
    descriptions = {
        "Grade 1": "Superficial Ulcer (full skin thickness, no subcutaneous involvement)",
        "Grade 2": "Deep Ulcer (penetrating to tendon, ligament, or capsule, without bone involvement)",
        "Grade 3": "Deep Ulcer with Abscess, Osteomyelitis, or Joint Sepsis",
        "Grade 4": "Localized Gangrene (forefoot or heel)"
    }
    for folder, idx in audit["folder_to_label_mapping"].items():
        desc = descriptions.get(folder, "Wagner Grade")
        lines.append(f"| `{folder}` | `{idx}` | {desc} |")
        
    lines.append("")
    lines.append("## Class Distribution Across Partitions")
    lines.append("| Split | Grade 1 (0) | Grade 2 (1) | Grade 3 (2) | Grade 4 (3) | Total |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for split, counts in sorted(audit["class_counts_by_split"].items()):
        g1 = counts.get("Grade 1", 0)
        g2 = counts.get("Grade 2", 0)
        g3 = counts.get("Grade 3", 0)
        g4 = counts.get("Grade 4", 0)
        tot = g1 + g2 + g3 + g4
        lines.append(f"| **{split}** | {g1:,} | {g2:,} | {g3:,} | {g4:,} | **{tot:,}** |")

    lines.append("")
    lines.append("## Integrity & Consistency Checks")
    lines.append(f"- **Unexpected Folders**: {len(audit['unexpected_folders'])}")
    lines.append(f"- **Hidden / OS System Files**: {len(audit['hidden_files'])}")
    lines.append(f"- **Empty Classes**: {len(audit['empty_classes'])}")
    lines.append(f"- **Duplicate Class Representations**: `{audit['duplicate_class_representations']}`")
    lines.append(f"- **Metadata Conflicts**: `{audit['metadata_conflict']}`")
    lines.append("")
    
    if audit["issues_found"]:
        lines.append("### Critical Issues Found")
        for err in audit["issues_found"]:
            lines.append(f"- ❌ {err}")
        lines.append("")
    else:
        lines.append("✅ **No critical label or folder structure issues found.**")
        lines.append("")
        
    if audit["warnings_found"]:
        lines.append("### Warnings")
        for w in audit["warnings_found"]:
            lines.append(f"- ⚠️ {w}")
        lines.append("")
        
    return "\n".join(lines)

def main():
    raw_dir = config.RAW_DATA
    print(f"Executing Label Verification on: {raw_dir}")
    
    audit_result = audit_labels(raw_dir)
    
    # Save outputs to metadata/ and interim/metadata/
    target_dirs = [
        config.METADATA_DIR,
        config.DATASET_ROOT / "interim" / "metadata"
    ]
    
    for t_dir in target_dirs:
        t_dir.mkdir(parents=True, exist_ok=True)
        json_path = t_dir / "label_audit.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(audit_result, f, indent=2)
        print(f"  [OK] Saved JSON audit: {json_path}")
        
        md_report = generate_markdown_report(audit_result)
        md_path = t_dir / "label_audit.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_report)
        print(f"  [OK] Saved Markdown audit report: {md_path}")
        
    print("\n==========================================")
    print(f"LABEL VERIFICATION STATUS: {audit_result['status']}")
    print("==========================================")
    try:
        print(generate_markdown_report(audit_result))
    except UnicodeEncodeError:
        print(generate_markdown_report(audit_result).encode("ascii", errors="replace").decode("ascii"))
        
    if audit_result["status"] != "PASS":
        sys.exit(1)

if __name__ == "__main__":
    main()
