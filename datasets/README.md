# FusionMedAI Datasets

This directory contains the datasets used in FusionMedAI for diabetes risk prediction across multiple modalities (retinal images, foot ulcer images, and clinical tabular data).

## Folder Structure

```directory
datasets/
├── raw/
│   └── aptos2019/
├── processed/
│   └── splits/
├── metadata/
│   ├── verification_report.json
│   ├── dataset_statistics.json
│   └── ...
├── interim/
└── README.md
```

## Dataset Overview

| Module | Dataset | Status |
| :--- | :--- | :--- |
| **Retina** | APTOS 2019 | ✅ Implemented |
| **Foot Ulcer** | DFUC | 🚧 Planned |
| **Clinical** | PIMA | 🚧 Planned |

---

# Dataset Setup

This project does not include the datasets due to licensing and size restrictions.

## APTOS 2019

1. Download the dataset from Kaggle:
   https://www.kaggle.com/competitions/aptos2019-blindness-detection

2. Extract it into:
   `datasets/raw/aptos2019/`

3. Run the pipeline setup and verification scripts:

```bash
# Step 1: Verify raw dataset integrity
python src/data/verify_dataset.py

# Step 2: Generate dataset metadata and statistics
python src/data/generate_metadata.py

# Step 3: Create stratified 80/10/10 dataset splits
python src/data/split_dataset.py

# Step 4: Verify complete end-to-end data pipeline
python src/data/verify_pipeline.py
```

## Retina Module Progress

Dataset Preparation          ✅
Data Pipeline               ✅
EDA                         ✅
Training Framework          ✅
Architecture Benchmarking   ✅
Calibration                 ⏳
Explainability              ⏳
Uncertainty                 ⏳

### Completed Tasks
✅ Dataset verification
✅ Metadata generation
✅ Stratified train/validation/test split
✅ Exploratory Data Analysis
✅ Baseline training framework
✅ Architecture benchmarking
✅ EfficientNet-B3 backbone selected

Current retinal backbone: **EfficientNet-B3**

## Reproducibility

The retinal experiments use:

- Fixed random seed
- Deterministic train/validation/test split
- Frozen benchmark configuration
- Versioned experiment outputs

## Benchmark Summary

Five pretrained architectures were evaluated under identical experimental settings:

- EfficientNet-B0
- EfficientNet-B3
- ConvNeXt-Tiny
- Swin-Tiny
- ViT-B/16

Selected retinal backbone: **EfficientNet-B3**

---

# Retina Module Dataset: APTOS 2019 Blindness Detection

### Overview
APTOS 2019 is a publicly available diabetic retinopathy dataset consisting of retinal fundus images collected from diabetic patients.

### Purpose
Diabetic Retinopathy Severity Classification

### Dataset Statistics
- **Dataset Version:** v1
- **Total Images:** 3,662 Training, 1,928 Testing
- **Image Format:** PNG

### Download Source
- **Link:** [Kaggle Competition](https://www.kaggle.com/competitions/aptos2019-blindness-detection)

### Labels
| Label | Severity |
|-------|----------|
| 0     | No DR |
| 1     | Mild DR |
| 2     | Moderate DR |
| 3     | Severe DR |
| 4     | Proliferative DR |

### License
According to Kaggle Competition Terms (Research/Educational Use)

---

## Planned Datasets

### Foot Ulcer
- **Dataset:** DFUC
- **Status:** Planned

### Clinical
- **Dataset:** PIMA Indians Diabetes
- **Status:** Planned

---

## Future Datasets

Additional external validation datasets (e.g., IDRiD, Messidor) may be incorporated in future research phases for evaluating model generalization.

## Dataset Notice

FusionMedAI does not redistribute any third-party datasets.

Users are responsible for downloading each dataset from its official source and complying with the respective licensing terms.