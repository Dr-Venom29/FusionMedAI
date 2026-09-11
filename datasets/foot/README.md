# Foot DFU Dataset (Wagner 4-Class Classification)

This directory contains the dataset artifacts, metadata, and directory structure for the **Diabetic Foot Ulcer (DFU)** modality of FusionMedAI.

## Directory Structure

```directory
datasets/foot/
├── raw/                         # Immutable original dataset source
│   ├── README.dataset.txt
│   ├── README.roboflow.txt
│   ├── train/
│   │   ├── Grade 1/
│   │   ├── Grade 2/
│   │   ├── Grade 3/
│   │   └── Grade 4/
│   ├── valid/
│   │   ├── Grade 1/
│   │   ├── Grade 2/
│   │   ├── Grade 3/
│   │   └── Grade 4/
│   └── test/
│       ├── Grade 1/
│       ├── Grade 2/
│       ├── Grade 3/
│       └── Grade 4/
├── interim/                     # Intermediate audit artifacts
│   └── metadata/
│       ├── inventory.json
│       └── dataset_inventory.md
├── processed/                   # Generated datasets/splits for downstream pipeline
│   └── splits/                  # (To be generated in Phase 10.2: train.csv, val.csv, test.csv)
└── metadata/                    # Persistent audit records
    ├── inventory.json
    ├── dataset_inventory.md
    ├── label_audit.json
    ├── label_audit.md
    ├── integrity_report.json
    ├── integrity_report.md
    ├── property_audit.json
    ├── property_audit.md
    ├── duplicate_report.json
    ├── duplicate_report.md
    ├── class_distribution_report.json
    ├── class_distribution_report.md
    ├── visual_quality_report.json
    ├── visual_quality_report.md
    ├── leakage_report.json
    ├── leakage_report.md
    ├── provenance_report.json
    ├── provenance_report.md
    ├── quality_decision_report.json
    ├── quality_decision_report.md
    ├── statistics/              # CSV index & tabular statistics
    │   ├── foot_image_inventory.csv
    │   ├── foot_image_properties.csv
    │   ├── foot_duplicate_pairs.csv
    │   ├── class_distribution.csv
    │   └── foot_leakage_groups.csv
    └── quality/                 # Visual contact sheets & quality metrics
        ├── wagner_contact_sheet.png
        └── quality_statistics.csv
```

---

## Dataset Inventory Summary

| Property | Value |
| :--- | :--- |
| **Dataset Name** | ADPM V3.3 Diabetic Foot Ulcer Classification |
| **Total Files** | 10,064 |
| **Total Images** | 10,062 |
| **Image Formats** | JPEG (`.jpg`) — 100% |
| **Color Space** | 3-Channel RGB — 100% |
| **Image Dimensions** | $224 \times 224$ pixels — 100% uniform |
| **Total Size on Disk** | 66.71 MB |
| **Average Image File Size** | 6.79 KB |
| **License** | MIT |

---

## Folder-to-Label Mapping

| Folder Name | Class Index | Clinical Description |
| :--- | :---: | :--- |
| `Grade 1` | `0` | Superficial Ulcer (full skin thickness, no subcutaneous involvement) |
| `Grade 2` | `1` | Deep Ulcer (penetrating to tendon, ligament, or capsule, without bone involvement) |
| `Grade 3` | `2` | Deep Ulcer with Abscess, Osteomyelitis, or Joint Sepsis |
| `Grade 4` | `3` | Localized Gangrene (forefoot or heel) |

---

## Class Distribution Across Partitions

| Partition Split | Grade 1 (`0`) | Grade 2 (`1`) | Grade 3 (`2`) | Grade 4 (`3`) | Total Images |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **train** | 2,240 | 2,345 | 2,744 | 2,310 | **9,639** |
| **valid** | 86 | 78 | 40 | 78 | **282** |
| **test** | 44 | 37 | 18 | 42 | **141** |
| **Total** | **2,370** | **2,460** | **2,802** | **2,430** | **10,062** |

---

## Verification Summary

```text
LABEL VERIFICATION STATUS: PASS
```

### 1. Explicit Folder-to-Label Mapping Verified

| Folder Name | Numerical Label | Wagner Grade & Clinical Description |
| :--- | :---: | :--- |
| `Grade 1` | `0` | Superficial Ulcer (full skin thickness, no subcutaneous involvement) |
| `Grade 2` | `1` | Deep Ulcer (penetrating to tendon, ligament, or capsule, without bone involvement) |
| `Grade 3` | `2` | Deep Ulcer with Abscess, Osteomyelitis, or Joint Sepsis |
| `Grade 4` | `3` | Localized Gangrene (forefoot or heel) |

### 2. Class Distribution Across Partitions

| Partition Split | Grade 1 (`0`) | Grade 2 (`1`) | Grade 3 (`2`) | Grade 4 (`3`) | Total Verified Images |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **train** | 2,240 | 2,345 | 2,744 | 2,310 | **9,639** |
| **valid** | 86 | 78 | 40 | 78 | **282** |
| **test** | 44 | 37 | 18 | 42 | **141** |
| **Total** | **2,370** | **2,460** | **2,802** | **2,430** | **10,062** |

### 3. Integrity & Consistency Audit Results

* **Unexpected Folders**: `0`
* **Hidden / OS System Files** (`.DS_Store`, `Thumbs.db`): `0`
* **Empty Classes**: `0`
* **Duplicate Class Representations** (e.g. `grade 1` vs `Grade_1`): `False`
* **Metadata Conflicts**: `False` (`README.dataset.txt` confirms 4 groups with MIT license)

### 4. Image Integrity Audit Results (`Phase 10.1.D`)

| Classification Status | Count | Percentage | Audit Verification Outcome |
| :--- | :---: | :---: | :--- |
| **VALID** | **10,062** | **100.00%** | Decodes cleanly, valid dimensions ($224 \times 224$), non-truncated |
| **CORRUPT** | `0` | `0.00%` | Zero corrupt or un-decodable images detected |
| **UNREADABLE** | `0` | `0.00%` | Zero 0-byte or unreadable files detected |
| **INVALID FORMAT** | `2` | N/A | Metadata text files (`README.dataset.txt`, `README.roboflow.txt`) |

### 5. Observed Dataset Properties & Candidate Preprocessing Options (`Phase 10.1.E`)

* **Observed Dataset Visual Statistics**:
  | Visual Metric | Red Channel | Green Channel | Blue Channel | Overall |
  | :--- | :---: | :---: | :---: | :---: |
  | **Observed Dataset Mean** | `0.4937` | `0.3630` | `0.3272` | `0.3946` |
  | **Observed Dataset Std** | `0.1744` | `0.1632` | `0.1551` | `0.1862` |

  ```python
  # Observed Foot DFU Dataset Normalization Values
  OBSERVED_DATASET_MEAN = [0.4937, 0.3630, 0.3272]
  OBSERVED_DATASET_STD  = [0.1744, 0.1632, 0.1551]
  ```

* **Candidate Preprocessing Options (To be evaluated in Step 10.2 - Data Pipeline)**:
  - **Target Resolution**: Native `$224 \times 224$` resolution (no aspect ratio distortion).
  - **Candidate Spatial Augmentations**: Rotation $\pm 15^\circ$, Horizontal Flip $p=0.5$.
  - **Candidate Color Jitter**: Brightness (0.2), Contrast (0.2), Saturation (0.1).
  - **Rationale**: Foot DFU images are $224 \times 224$ RGB clinical wound photos. We preserve the scientific separation between measured dataset observations and pipeline candidate choices evaluated during Step 10.2.

### 6. Duplicate & Leakage Audit Findings (`Phase 10.1.F`)

| Audit Check | Count Detected | Status | Clinical & Pipeline Impact |
| :--- | :---: | :---: | :--- |
| **Unique Exact Images (SHA-256)** | **10,050** | ✅ PASS | 12 exact duplicate groups (24 total files) |
| **Exact Cross-Split Data Leakage** | `0` | ✅ PASS | Zero byte-exact images shared between train and test/valid splits |
| **Exact Cross-Class Label Conflict** | `0` | ✅ PASS | Zero byte-exact images assigned to multiple Wagner grades |
| **Near Cross-Split Leakage (dHash $\le 4$)** | `5` pairs | ⚠️ Flagged | 5 augmented variants found across train/valid/test splits |
| **Near Cross-Class Conflict (dHash $\le 4$)** | `114` pairs | ⚠️ Flagged | 114 augmented variants assigned across different Wagner grades |

### 7. Class Distribution & Imbalance Audit Findings (`Phase 10.1.G`)

| Wagner Grade | Class Index | Total Images | Percentage | Distribution Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **Grade 1** | `0` | 2,370 | **23.55%** | Minority Class |
| **Grade 2** | `1` | 2,460 | **24.45%** | Balanced |
| **Grade 3** | `2` | 2,802 | **27.85%** | Majority Class |
| **Grade 4** | `3` | 2,430 | **24.15%** | Balanced |
| **Total** | | **10,062** | **100.00%** | **Overall Imbalance Ratio: `1.18 : 1` (WELL_BALANCED)** |

### 8. Data Leakage & Patient-Case Investigation (`Phase 10.1.I`)

| Leakage Metric | Finding | Evaluation Impact & Required Action |
| :--- | :---: | :--- |
| **Total Image Files** | **10,062** | 10,062 total `.jpg` files in `datasets/foot/raw/` |
| **Unique Source Images** | **1,770** | Source images expanded $5.68\times$ via offline Roboflow augmentations |
| **Source Image Cross-Split Leakage** | `10` groups | ⚠️ Augmented variants of 10 source images exist across `train` and `test`/`valid` |
| **Source Image Cross-Class Conflict** | `23` groups | ⚠️ Augmented variants of 23 source images assigned across different Wagner grades |
| **Patient Identifier Availability** | `UNAVAILABLE` | **Documented Limitation**: Patient IDs are not provided in raw metadata. Separation beyond source-image grouping cannot be guaranteed. |

### 9. Provenance & License Audit Findings (`Phase 10.1.J`)

| Provenance Attribute | Finding / Detail |
| :--- | :--- |
| **Access Platform Host** | Kaggle (`DFU_Dataset_annotated_into_4_classes`) |
| **Roboflow Universe Project** | [ADPM V3.3 Classification](https://universe.roboflow.com/adpm/adpm-v3.3-classification) (Version 4, exported April 5, 2025) |
| **Curator / Uploader Account** | Roboflow User `adpm` |
| **Declared License** | **MIT License** |
| **Kaggle Uploader vs Creator** | Kaggle repository is a community mirror; original curation and offline augmentation were exported from Roboflow Universe by account `adpm`. |
| **License vs Clinical Provenance** | The declared MIT license applies to the distributed dataset export. The underlying clinical-image provenance and original photography rights are not fully documented in the distributed metadata. |
| **Medical Annotation Claims** | Classified into Wagner–Meggitt Grades 1–4. Individual clinician identities and inter-rater reliability scores ($\kappa$) are not provided in metadata. |

* **Citation Requirements**:
  ```bibtex
  @misc{adpm_dfu_2025,
    title={ADPM V3.3 Classification Dataset},
    author={adpm},
    year={2025},
    publisher={Roboflow Universe},
    url={https://universe.roboflow.com/adpm/adpm-v3.3-classification}
  }

  @article{wagner1981dysvascular,
    title={The dysvascular foot: a system for diagnosis and treatment},
    author={Wagner, F William},
    journal={Foot & Ankle},
    volume={2},
    number={2},
    pages={64--122},
    year={1981}
  }
  ```

---

## Dataset Limitations

- **Patient-level identifiers are unavailable** in the distributed dataset metadata.
- The dataset contains **1,770 source-image groups** expanded into **10,062 image files** via offline Roboflow augmentations.
- **Near-duplicate analysis** identified augmented variants crossing the original `train`/`valid`/`test` raw partitions.
- Therefore, the original dataset raw folder partitions are **not** accepted as-is for downstream modeling.
- **Phase 10.2 must construct new group-stratified partitions** using `source_image_id` to prevent cross-split data leakage.
- **Patient-level separation cannot be guaranteed** because patient-level identifiers are unavailable in the distributed dataset metadata.

---

## 10. Formal Dataset Quality Decision (`Phase 10.1.K`)

```text
================================================================================
FINAL QUALITY DECISION: CONDITIONAL PASS
Status: APPROVED WITH MANDATORY GROUP-STRATIFIED SPLITTING REQUIREMENTS
Dataset Freeze: FROZEN & VERIFIED (Phase 10.1 Completed)
================================================================================
```

> [!IMPORTANT]
> **Dataset Freeze Definition:**  
> The raw dataset contents (`datasets/foot/raw/`) and observed audit evidence are frozen.  
> The original `train`/`valid`/`test` directory assignments are **not** accepted as the final modeling split. Phase 10.2 will construct leakage-controlled, source-group-stratified splits from the frozen raw dataset.

| Decision Metric | Result & Policy |
| :--- | :--- |
| **Formal Decision** | **CONDITIONAL PASS** |
| **Decision Rationale** | Image integrity (100% decodable), label alignment, class distribution (1.18:1), and visual clarity are high. However, because 1,770 source images were expanded $5.68\times$ into 10,062 image files via offline augmentations without patient IDs, random splitting causes cross-split leakage. |
| **Mandatory Condition 1** | **PROHIBIT** random image-level splitting across training, validation, and test partitions. |
| **Mandatory Condition 2** | **ENFORCE** `source_image_id` group-stratified splitting in Phase 10.2 (`split_dataset.py`) to achieve 0% cross-split leakage. |
| **Mandatory Condition 3** | **DOCUMENT** patient ID unavailability limitation in research documentation. |
| **Dataset Freeze Status** | **FROZEN** as of September 11, 2026. Ready for Phase 10.2 Data Pipeline development. |

---

> **Immutable Raw Source Rule**:
> The contents of `datasets/foot/raw/` are treated as strictly **read-only**.
> No raw image resizing, renaming, duplicate deletion, label editing, normalization, or augmentation is performed directly inside `raw/`. All preprocessing occurs programmatically in `interim/` and `processed/`.
