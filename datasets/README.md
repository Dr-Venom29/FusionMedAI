# FusionMedAI Datasets

This directory contains the datasets used in FusionMedAI for diabetes risk prediction across multiple modalities (retinal images, foot ulcer images, and clinical tabular data).

## Folder Structure

```directory
datasets/
├── raw/
│   └── aptos2019/
├── interim/          # Intermediate files generated during preprocessing
├── processed/        # Versioned train/val/test splits
├── metadata/         # Generated metadata and statistics
└── README.md
```

## Dataset Overview

| Module | Dataset | Purpose |
| :--- | :--- | :--- |
| **Retina** | APTOS 2019 | Diabetic Retinopathy Classification |
| **Foot Ulcer** | DFUC | Wound Detection & Staging |
| **Clinical** | PIMA | Tabular Diabetes Risk Prediction |

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

# Step 3: Create stratified dataset splits (80/10/10)
python src/data/split_dataset.py

# Step 4: Verify complete end-to-end data pipeline
python src/data/verify_pipeline.py
```

## Current Status

The APTOS 2019 dataset has completed:
- Dataset verification
- Metadata generation
- Stratified train/validation/test splitting
- Exploratory Data Analysis (EDA)
- Baseline framework validation

The dataset is now ready for baseline model training.

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

# Foot Ulcer Module Dataset: DFUC (Diabetic Foot Ulcer Challenge)

### Overview
DFUC is a diabetic foot ulcer image dataset used for ulcer detection, localization, segmentation, and severity assessment.

### Dataset Statistics
- **Approximate Image Count:** ~4,000+ Images

### Download Source
- **Link:** [DFUC Challenge Website](https://dfu-challenge.github.io)

### Classes
Depending on the specific task:
- Ulcer Present / Ulcer Absent
- Severity Categories

### License
Research Use

---

# Clinical Module Dataset: PIMA Indians Diabetes Dataset

### Overview
PIMA is a structured clinical dataset used for diabetes prediction based on diagnostic measurements.

### Dataset Statistics
- **Sample Count:** 768 Patients
- **Features:** 8 Clinical Features:
  1. Pregnancies
  2. Glucose
  3. Blood Pressure
  4. Skin Thickness
  5. Insulin
  6. BMI
  7. Diabetes Pedigree Function
  8. Age

### Download Source
- **Link:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)

### Target Labels
| Label | Meaning |
|-------|----------|
| 0     | Non-Diabetic |
| 1     | Diabetic |

### Missing Values (Known Issues)
Several features contain zero values that represent missing measurements. These include:
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI

### License
Open Research Dataset

---

## Future Datasets

Additional external validation datasets (e.g., IDRiD, Messidor) may be incorporated in future research phases for evaluating model generalization.

## Dataset Notice

FusionMedAI does not redistribute any third-party datasets.

Users are responsible for downloading each dataset from its official source and complying with the respective licensing terms.