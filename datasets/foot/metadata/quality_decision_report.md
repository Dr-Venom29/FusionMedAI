# Phase 10.1.K — Dataset Quality Decision Report

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
| **10.1.C** | Label Verification | Folder mapping Grade 1..4 $\rightarrow$ 0..3 verified | **PASS** |
| **10.1.D** | Image Integrity | 10,062 VALID decodable images (0 corrupt, 0 unreadable) | **PASS** |
| **10.1.E** | Image Properties | Observed Mean `[0.4937, 0.3630, 0.3272]`, Std `[0.1744, 0.1632, 0.1551]` | **PASS** |
| **10.1.F** | Duplicate Detection | 10,050 unique SHA-256 hashes, 12 exact duplicate groups | **PASS** |
| **10.1.G** | Class Distribution | Imbalance ratio `1.18 : 1` (WELL_BALANCED across all 4 grades) | **PASS** |
| **10.1.H** | Visual Quality | High clinical clarity, 0 non-foot images, contact sheet generated | **PASS** |
| **10.1.I** | Data Leakage | 1,770 source images expanded $5.68\times$; 10 raw subfolder leakages | **CONDITIONAL** |
| **10.1.J** | Provenance & License | Kaggle mirror of Roboflow Universe `ADPM V3.3`, MIT License | **PASS** |

---

## 2. Decision Rationale & Mandatory Conditions

### Rationale
The Diabetic Foot Ulcer (DFU) Wagner 4-Class dataset demonstrates exceptional image decodability, clean folder-to-class alignment, well-balanced class balance, high visual clarity, and transparent MIT licensing. However, because 1,770 source images were expanded $5.68\times$ into 10,062 image files via offline Roboflow augmentations, naive random splitting or relying on raw subfolders causes cross-split data leakage.

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
