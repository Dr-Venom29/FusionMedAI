# FusionMedAI Dataset Verification

## Purpose

This document verifies the datasets used in FusionMedAI and records important information regarding dataset characteristics, labels, licensing, and suitability for the proposed research.

---

# Dataset 1: APTOS 2019 Blindness Detection

## Overview

APTOS 2019 is a publicly available diabetic retinopathy dataset consisting of retinal fundus images collected from diabetic patients.

## Intended Use

### Retinal Risk Assessment Module

Tasks:

- Diabetic Retinopathy Detection
- Retinopathy Severity Prediction
- Grad-CAM Explainability
- Confidence Estimation
- Uncertainty Estimation

## Dataset Details

| Attribute | Value |
|------------|---------|
| Dataset Name | APTOS 2019 Blindness Detection |
| Modality | Retinal Fundus Images |
| Data Type | Image |
| Image Count | 3,662 Images |
| Resolution | Variable |
| Source | Kaggle |
| License | Competition Dataset (Research/Educational Use) |

## Labels

| Label | Severity |
|---------|----------|
| 0 | No DR |
| 1 | Mild DR |
| 2 | Moderate DR |
| 3 | Severe DR |
| 4 | Proliferative DR |

## Verification Checklist

- [ ] Dataset Downloaded
- [ ] Image Count Verified
- [ ] Labels Verified
- [ ] Class Distribution Analyzed
- [ ] Train / Validation / Test Split Created
- [ ] Data Leakage Checked

## Download Link

https://www.kaggle.com/competitions/aptos2019-blindness-detection

---

# Dataset 2: DFUC (Diabetic Foot Ulcer Challenge Dataset)

## Overview

DFUC is a diabetic foot ulcer image dataset used for ulcer detection, localization, segmentation, and severity assessment.

## Intended Use

### Foot Risk Assessment Module

Tasks:

- Foot Ulcer Detection
- Foot Ulcer Severity Prediction
- Grad-CAM Explainability
- Confidence Estimation
- Uncertainty Estimation

## Dataset Details

| Attribute | Value |
|------------|---------|
| Dataset Name | DFUC |
| Modality | Foot Ulcer Images |
| Data Type | Image |
| Approximate Image Count | ~4,000+ Images |
| Source | DFUC Challenge |
| License | Research Use |

## Classes

Depending on task:

- Ulcer Present
- Ulcer Absent

or

- Severity Categories

(To be verified after download)

## Verification Checklist

- [ ] Dataset Downloaded
- [ ] Image Count Verified
- [ ] Class Labels Verified
- [ ] Class Distribution Analyzed
- [ ] Train / Validation / Test Split Created
- [ ] Data Leakage Checked

## Download Link

https://dfu-challenge.github.io

---

# Dataset 3: PIMA Indians Diabetes Dataset

## Overview

PIMA is a structured clinical dataset used for diabetes prediction based on diagnostic measurements.

## Intended Use

### Clinical Risk Assessment Module

Tasks:

- Clinical Diabetes Risk Prediction
- SHAP Explainability
- Confidence Estimation
- Uncertainty Estimation

## Dataset Details

| Attribute | Value |
|------------|---------|
| Dataset Name | PIMA Indians Diabetes Dataset |
| Modality | Clinical Tabular Data |
| Data Type | Structured Data |
| Sample Count | 768 Patients |
| Features | 8 Clinical Features |
| Source | UCI Machine Learning Repository |
| License | Open Research Dataset |

## Features

1. Pregnancies
2. Glucose
3. Blood Pressure
4. Skin Thickness
5. Insulin
6. BMI
7. Diabetes Pedigree Function
8. Age

## Target Labels

| Label | Meaning |
|---------|----------|
| 0 | Non-Diabetic |
| 1 | Diabetic |

## Missing Values

Known Issue:

Several features contain zero values that represent missing measurements.

Affected Features:

- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI

Missing-value handling strategy must be documented during preprocessing.

## Verification Checklist

- [ ] Dataset Downloaded
- [ ] Feature Definitions Verified
- [ ] Missing Values Analyzed
- [ ] Data Cleaning Performed
- [ ] Train / Validation / Test Split Created
- [ ] Data Leakage Checked

## Download Link

https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes

---

# Dataset Alignment Statement

## Important Research Assumption

FusionMedAI does **NOT** perform patient-level multimodal learning.

Datasets are independent and originate from different patient populations.

Each modality-specific model is trained independently:

```text
APTOS
→ Retinal Risk

DFUC
→ Foot Risk

PIMA
→ Clinical Risk
```

ACARA-U performs **decision-level aggregation** using:

- Risk Scores
- Confidence Scores
- Reliability Scores
- Uncertainty Scores

No raw patient features are fused across datasets.

---

# Dataset Risks

## Risk 1: Independent Datasets

Issue:

Datasets do not contain the same patients.

Mitigation:

Decision-level aggregation through ACARA-U.

---

## Risk 2: Class Imbalance

Issue:

Certain classes may have significantly fewer samples.

Mitigation:

- Class weighting
- Data augmentation
- Stratified splitting

---

## Risk 3: Domain Shift

Issue:

Data distributions may differ across datasets.

Mitigation:

- OOD Detection Module
- Calibration Engine
- Reliability Learning

---

## Risk 4: Low-Quality Inputs

Issue:

Blurred images, poor contrast, incomplete reports.

Mitigation:

- Data Quality Assessment Module
- Quality Score Computation

---

# Dataset Status

| Dataset | Status |
|----------|----------|
| APTOS 2019 | Pending Download |
| DFUC | Pending Download |
| PIMA | Pending Download |

---

# Notes

Before implementation:

- Verify exact DFUC version (2020 / 2021 / 2022)
- Verify licensing requirements
- Verify image counts after download
- Document train/validation/test splits
- Document preprocessing pipeline

---

**Last Updated:** YYYY-MM-DD
**Project:** FusionMedAI
**Version:** Dataset Verification v1