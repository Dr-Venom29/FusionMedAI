# FusionMedAI Datasets

This directory contains the datasets used in FusionMedAI for diabetes risk prediction across multiple modalities (retinal images, foot ulcer images, and clinical tabular data).

## Folder Structure

```text
datasets/
├── raw/
│   └── aptos2019/
│       ├── train_images/
│       ├── test_images/
│       ├── train.csv
│       ├── test.csv
│       └── sample_submission.csv
├── processed/
├── interim/
└── metadata/
```

---

# Dataset Setup

This project does not include the datasets due to licensing and size restrictions.

## APTOS 2019

1. Download the dataset from Kaggle:
   https://www.kaggle.com/competitions/aptos2019-blindness-detection

2. Extract it into:

```
datasets/raw/aptos2019/
```

The directory should look like:

```
datasets/
└── raw/
    └── aptos2019/
        ├── train_images/
        ├── test_images/
        ├── train.csv
        ├── test.csv
        └── sample_submission.csv
```

3. Verify the dataset:

```bash
python src/data/verify_dataset.py
```

4. Generate metadata:

```bash
python src/data/generate_metadata.py
```

---

# Dataset 1: APTOS 2019 Blindness Detection

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

# Dataset 2: DFUC (Diabetic Foot Ulcer Challenge Dataset)

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

# Dataset 3: PIMA Indians Diabetes Dataset

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