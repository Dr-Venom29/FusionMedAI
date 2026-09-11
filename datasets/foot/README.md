# Foot DFU Dataset (Wagner 4-Class Classification)

This directory contains the dataset artifacts, metadata, and directory structure for the **Diabetic Foot Ulcer (DFU)** modality of FusionMedAI.

## Directory Structure

```directory
datasets/foot/
├── raw/                         # Immutable original dataset source (Read-only)
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
├── processed/                   # Generated canonical dataset and splits
│   └── splits/                  # train.csv, val.csv, test.csv, index.csv
└── metadata/                    # Persistent audit records & EDA outputs
    ├── eda/                     # Phase 10.3 Exploratory Data Analysis Outputs
    │   ├── statistics/          # Tabular stats & profiling JSON
    │   │   ├── foot_eda_stats.csv
    │   │   ├── class_distribution.csv
    │   │   ├── split_distribution.csv
    │   │   └── eda_statistical_profiling.json
    │   ├── quality/             # Canonical quality metrics & outlier manifests
    │   │   ├── quality_metrics_canonical.csv
    │   │   ├── quality_outliers.csv
    │   │   └── outlier_analysis_manifest.csv
    │   ├── figures/             # Visual grid contact sheets & heatmaps
    │   │   ├── wagner_class_comparison_grid.png
    │   │   ├── class_separability_pca_tsne.png
    │   │   └── bias_correlation_matrix.png
    │   └── reports/             # Modular EDA research reports & final decision
    │       ├── eda_statistical_profiling.md
    │       ├── class_visual_analysis.json
    │       ├── eda_image_quality.json
    │       ├── eda_image_quality.md
    │       ├── eda_outlier_analysis.json
    │       ├── eda_outlier_analysis.md
    │       ├── eda_class_separability.json
    │       ├── eda_class_separability.md
    │       ├── eda_dataset_bias.json
    │       ├── eda_dataset_bias.md
    │       ├── eda_sampling_strategy.json
    │       ├── eda_sampling_strategy.md
    │       ├── eda_final_decision.json
    │       └── eda_final_decision.md
    ├── statistics/              # CSV index & tabular statistics
    └── quality/                 # Visual contact sheets & quality metrics
```

---

## Dataset Inventory Summary

| Property | Value |
| :--- | :--- |
| **Dataset Name** | ADPM V3.3 Diabetic Foot Ulcer Classification |
| **Total Raw Files** | 10,064 (10,062 JPEG images + 2 text metadata files) |
| **Canonical Modeling Population** | 10,050 images (12 exact duplicates excluded) |
| **Source Image Groups** | 1,770 source groups |
| **Image Formats** | JPEG (`.jpg`) — 100% |
| **Color Space** | 3-Channel RGB — 100% |
| **Image Dimensions** | $224 \times 224$ pixels — 100% uniform |
| **License** | MIT License |

---

## Folder-to-Label Mapping

| Folder Name | Class Index | Clinical Description |
| :--- | :---: | :--- |
| `Grade 1` | `0` | Superficial Ulcer (full skin thickness, no subcutaneous involvement) |
| `Grade 2` | `1` | Deep Ulcer (penetrating to tendon, ligament, or capsule, without bone involvement) |
| `Grade 3` | `2` | Deep Ulcer with Abscess, Osteomyelitis, or Joint Sepsis |
| `Grade 4` | `3` | Localized Gangrene (forefoot or heel) |

---

## Dataset Construction & Modeling Splits (Phase 10.2)

The audited raw dataset contains 10,062 valid JPEG RGB images at $224 \times 224$ resolution. Following the Phase 10.1 audit, 10,050 canonical images were retained for downstream modeling. Exact duplicate handling, source-image grouping, near-duplicate analysis, and group-stratified splitting were completed during Phase 10.2.

The final modeling population consists of:
- **8,038 training images** (80%)
- **1,006 validation images** (10%)
- **1,006 test images** (10%)
- **1,770 source-image groups**

Zero source-group overlap was verified across the final modeling splits. Patient-level separation cannot be independently verified because patient identifiers are unavailable in the distributed dataset metadata.

---

## Exploratory Data Analysis & Dataset Quality (Phase 10.3)

Phase 10.3 examined the canonical modeling population across statistical, visual, image-quality, outlier, class-separability, and dataset-bias dimensions.

The analysis found a relatively balanced four-class distribution (Imbalance Ratio 1.18:1) and identified substantial visual overlap between Grades 2 and 3. Image-quality variation and capture-related artifacts were also documented rather than removed from the dataset.

No samples were deleted as a result of the outlier analysis.

The separability and shortcut analyses are used as diagnostic evidence for subsequent model development rather than as evidence of model performance.

Detailed analyses and generated artifacts are maintained under `research/foot/` and `datasets/foot/metadata/eda/`.

---

## Foot Ulcer Research & Pipeline Progression

| Phase | Topic | Status |
| :--- | :--- | :---: |
| **10.1** | Dataset Preparation & Audit | ✅ **Conditional Pass** |
| **10.2** | Data Pipeline | ✅ **PASS** |
| **10.3.1** | Dataset Statistical Profiling | ✅ **PASS** |
| **10.3.2** | Class-Wise Visual Analysis | ✅ **PASS** |
| **10.3.3** | Image Quality Analysis | ✅ **PASS** |
| **10.3.4** | Outlier Analysis | ✅ **PASS** |
| **10.3.5** | Class Separability Analysis | ✅ **PASS** |
| **10.3.6** | Dataset Bias & Shortcut Analysis | ✅ **PASS** |
| **10.4** | Baseline Framework | ⬜ **NEXT** |
| **10.5** | Architecture Benchmarking | ⬜ |
| **10.6** | Explainability | ⬜ |
| **10.7** | Probability Calibration | ⬜ |
| **10.8** | Uncertainty Estimation | ⬜ |
| **10.9** | Module Integration | ⬜ |
