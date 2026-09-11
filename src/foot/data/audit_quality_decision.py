import os
import sys
import json
from pathlib import Path

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.foot.config import DATASET_ROOT, RAW_DATA, METADATA_DIR

def run_quality_decision_audit():
    print("==================================================")
    print("Running Phase 10.1.K — Dataset Quality Decision Audit")
    print("==================================================")
    
    decision_data = {
        "audit_phase": "10.1.K",
        "dataset_name": "Diabetic Foot Ulcer (DFU) Wagner 4-Class Dataset",
        "overall_decision": "CONDITIONAL PASS",
        "decision_rationale": (
            "The dataset exhibits 100% valid image integrity, clear folder-to-label mapping, "
            "manageable duplicates, acceptable visual quality, well-balanced class distribution (1.18:1), "
            "and documented MIT provenance. However, because 1,770 source images were expanded 5.68x via "
            "offline augmentations without patient-level IDs, random splitting causes cross-split leakage. "
            "The dataset is CONDITIONALLY APPROVED for model research subject to mandatory group-stratified splitting on source_image_id."
        ),
        "subphase_audit_summary": {
            "10.1.B_inventory": "PASS — 10,062 images, 224x224 RGB JPEGs, 66.71 MB.",
            "10.1.C_label_verification": "PASS — Grade 1..4 mapped to numerical labels 0..3.",
            "10.1.D_integrity_audit": "PASS — 10,062 VALID decodable images (0 corrupt, 0 unreadable).",
            "10.1.E_property_audit": "PASS — Observed Mean [0.4937, 0.3630, 0.3272], Std [0.1744, 0.1632, 0.1551].",
            "10.1.F_duplicate_detection": "PASS — 10,050 unique SHA-256 hashes, 12 exact duplicate groups.",
            "10.1.G_class_distribution": "PASS — Well-balanced (imbalance ratio 1.18:1; G1: 23.55%, G2: 24.45%, G3: 27.85%, G4: 24.15%).",
            "10.1.H_visual_quality": "PASS — High-resolution clinical wound photos, 0 non-foot images, contact sheet archived.",
            "10.1.I_leakage_investigation": "CONDITIONAL — 1,770 source image groups identified; patient IDs unavailable; 10 cross-split leakage instances in raw subfolders requiring group-stratified re-partitioning.",
            "10.1.J_provenance_audit": "PASS — Kaggle mirror of Roboflow Universe 'ADPM V3.3', MIT License, bibtex citations documented."
        },
        "mandatory_conditions": [
            "1. STRICTLY PROHIBIT random image-level splitting across training, validation, and testing.",
            "2. ENFORCE source_image_id group-stratified splitting in Phase 10.2 (split_dataset.py) to achieve 0% cross-split leakage.",
            "3. EXPLICITLY DOCUMENT patient-level ID unavailability limitation in research papers and technical reports.",
            "4. KEEP datasets/foot/raw/ strictly immutable (read-only)."
        ],
        "dataset_freeze_status": {
            "frozen": True,
            "freeze_date": "2026-09-11",
            "target_modality": "Foot (DFU Wagner 4-Class)",
            "pipeline_readiness": "READY FOR PHASE 10.2 DATA PIPELINE DEVELOPMENT"
        }
    }
    
    # Save JSON report
    json_path = METADATA_DIR / "quality_decision_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(decision_data, f, indent=2)
    print(f"Saved JSON decision report to: {json_path}")
    
    # Generate Markdown report
    md_content = f"""# Phase 10.1.K — Dataset Quality Decision Report

## Executive Summary & Formal Decision

```text
================================================================================
FINAL QUALITY DECISION: CONDITIONAL PASS
Status: APPROVED WITH MANDATORY GROUP-STRATIFIED SPLITTING REQUIREMENTS
Dataset Freeze: PROCEEDED AND FROZEN (Phase 10.1 Completed)
================================================================================
```

---

## 1. Sub-Phase Synthesis Matrix

| Sub-Phase | Focus Area | Audit Findings | Status |
| :--- | :--- | :--- | :---: |
| **10.1.B** | Dataset Inventory | 10,062 images (100% 224x224 RGB JPEG, 66.71 MB) | **PASS** |
| **10.1.C** | Label Verification | Folder mapping Grade 1..4 $\\rightarrow$ 0..3 verified | **PASS** |
| **10.1.D** | Image Integrity | 10,062 VALID decodable images (0 corrupt, 0 unreadable) | **PASS** |
| **10.1.E** | Image Properties | Observed Mean `[0.4937, 0.3630, 0.3272]`, Std `[0.1744, 0.1632, 0.1551]` | **PASS** |
| **10.1.F** | Duplicate Detection | 10,050 unique SHA-256 hashes, 12 exact duplicate groups | **PASS** |
| **10.1.G** | Class Distribution | Imbalance ratio `1.18 : 1` (WELL_BALANCED across all 4 grades) | **PASS** |
| **10.1.H** | Visual Quality | High clinical clarity, 0 non-foot images, contact sheet generated | **PASS** |
| **10.1.I** | Data Leakage | 1,770 source images expanded $5.68\\times$; 10 raw subfolder leakages | **CONDITIONAL** |
| **10.1.J** | Provenance & License | Kaggle mirror of Roboflow Universe `ADPM V3.3`, MIT License | **PASS** |

---

## 2. Decision Rationale & Mandatory Conditions

### Rationale
The Diabetic Foot Ulcer (DFU) Wagner 4-Class dataset demonstrates exceptional image decodability, clean folder-to-class alignment, well-balanced class balance, high visual clarity, and transparent MIT licensing. However, because 1,770 source images were expanded $5.68\\times$ into 10,062 image files via offline Roboflow augmentations, naive random splitting or relying on raw subfolders causes cross-split data leakage.

### Mandatory Downstream Conditions (`Phase 10.2`)
1. **No Random Splitting**: Random image-level partitioning is strictly prohibited.
2. **Group-Stratified Partitioning**: `src/foot/data/split_dataset.py` must group all augmented variants sharing the same `source_image_id` into the same split (`train`, `val`, or `test`), ensuring **0% group leakage**.
3. **Patient ID Limitation**: Document the absence of patient-level IDs in research publications.
4. **Raw Immutability**: `datasets/foot/raw/` remains strictly read-only.

---

## 3. Dataset Freeze Declaration

The **Foot DFU Wagner 4-Class Dataset** is officially **FROZEN** as of **September 11, 2026**.

- **Modality**: Foot (DFU Wagner 4-Class Classification)
- **Raw Root**: `datasets/foot/raw/`
- **Freeze Status**: `FROZEN & VERIFIED`
- **Next Action**: Proceed directly to **Phase 10.2 — Foot Data Pipeline & Group-Based Partitioning**.
"""

    md_path = METADATA_DIR / "quality_decision_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved Markdown decision report to: {md_path}")
    
    print("\nPhase 10.1.K Dataset Quality Decision completed successfully.")

if __name__ == "__main__":
    run_quality_decision_audit()
