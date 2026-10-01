# Dataset Verification

## Overview

FusionMedAI uses separate datasets for the Retina, Foot Ulcer, and Clinical modalities. Dataset verification is performed independently for each modality before model development and evaluation.

The datasets are not treated as a single patient-level cohort. Their independent provenance, population, feature spaces, and acquisition characteristics are preserved throughout the modality-specific research pipelines.

---

# Dataset Verification Checklist

## Dataset 1: APTOS 2019 Blindness Detection

### Dataset Integrity
- [x] Dataset downloaded
- [x] Image count verified ($N=3,662$)
- [x] Image resolution verified
- [x] File integrity verified
- [x] Corrupted images checked
- [x] Missing files checked

### Labels and Distribution
- [x] Labels verified (5 diabetic retinopathy severity grades)
- [x] Class distribution analyzed
- [x] Duplicate images audited
- [x] Image quality assessed

### Data Pipeline
- [x] Metadata generated
- [x] Stratified train / validation / test split created ($80/10/10$)
- [x] Data leakage checked
- [x] Dataset class verified
- [x] Transform pipeline verified
- [x] DataLoader verified
- [x] End-to-end pipeline verified

### Downstream Retina Evaluation
- [x] Baseline framework verified
- [x] Baseline training completed
- [x] Architecture benchmarking completed (EfficientNet-B3 selected)
- [x] Explainability completed (Grad-CAM)
- [x] Probability calibration completed (Temperature Scaling)
- [x] Uncertainty estimation completed (MC Dropout)

---

## Dataset 2: ADPM V3.3 Diabetic Foot Ulcer Classification

### Dataset Integrity
- [x] Dataset inventory completed ($N=10,062$ audited images)
- [x] Image count verified
- [x] Image resolution verified
- [x] File integrity verified
- [x] Corrupted images checked
- [x] Missing files checked

### Labels and Dataset Quality
- [x] Class labels verified (4 Wagner grades)
- [x] Class distribution analyzed
- [x] Exact duplicates audited
- [x] Near-duplicate relationships analyzed
- [x] Source-image groups identified ($1,770$ patient clusters)
- [x] Image quality assessed
- [x] Outliers analyzed
- [x] Dataset bias and shortcut analysis completed

### Data Pipeline
- [x] Canonical manifest generated
- [x] Source-group split created (Train: 8,038, Val: 1,006, Held-out Test: 1,006)
- [x] Data leakage checks completed (Zero group-leakage)
- [x] Transform pipeline verified
- [x] DataLoader verified
- [x] End-to-end pipeline verified

### Downstream Foot Ulcer Evaluation
- [x] Baseline framework completed (ResNet-50)
- [x] Baseline training completed
- [x] Architecture benchmarking completed (EfficientNet-B3 selected)
- [x] Explainability completed (Grad-CAM + randomization checks)
- [x] Probability calibration completed (Vector Scaling)
- [x] Uncertainty estimation completed (MC Dropout + risk-coverage)

---

## Dataset 3: Clinical Risk Dataset (UCI Diabetes 130-US Hospitals)

### Dataset Preparation and Verification
- [x] Dataset acquired ($101,766$ encounters across 130 hospitals, 1999–2008)
- [x] Feature definitions verified
- [x] Missing values analyzed
- [x] Data cleaning and patient deduplication completed ($N=99,343$)
- [x] 119-dimensional feature representation established
- [x] Patient-level canonical train / validation / test splits created (Train: 69,519, Val: 14,911, Locked Test: 14,913)
- [x] Data leakage checks completed (Zero patient overlap across partitions)
- [x] Locked clinical preprocessor verified

### Clinical Research Pipeline
The Clinical modality has subsequently progressed through:
- [x] Model selection & validation-only HPO (CatBoost selected)
- [x] Explainability & stability auditing (Exact TreeSHAP, $\rho = 0.9994$)
- [x] Probability calibration & decision curve analysis (Isotonic Regression)
- [x] Prediction uncertainty estimation (50-member Bootstrap Ensemble)
- [x] Robustness and distribution-shift analysis (11 shift scenarios audited)
- [x] Clinical inference integration (Standardized `ClinicalOutput` schema)
- [x] End-to-end validation (10/10 verification gate passed)

---

# Dataset Status

| Dataset | Modality | Status | Partitions |
| :--- | :--- | :--- | :--- |
| **APTOS 2019** | Retina | Complete | Train (2,929), Val (366), Test (367) |
| **ADPM V3.3** | Foot Ulcer | Complete | Train (8,038), Val (1,006), Test (1,006) |
| **UCI Diabetes (130-US)** | Clinical | Complete (C1–C10) | Train (69,519), Val (14,911), Test (14,913) |

The three datasets remain independently maintained. Completion of the individual modality pipelines does not imply that the datasets can be merged into a common patient-level cohort.

---

# Dataset-Level Risks & Methodological Mitigations

## Independent Patient Populations
- **Issue**: The public datasets originate from different populations and do not provide a common patient identifier or shared longitudinal cohort.
- **Consequence**: Direct patient-level feature merging is unsupported and scientifically invalid.
- **Planned Research Approach**: Multimodal integration will operate on modality-level outputs rather than attempting to construct a synthetic patient-level merged dataset. This will be investigated in **ACARA-U Multimodal Fusion**.

---

## Domain and Acquisition Shift
- **Issue**: Different datasets exhibit distinct acquisition conditions, devices, clinical protocols, and population demographics.
- **Existing Mitigations**:
  - Modality-specific preprocessing contracts
  - Independent modality evaluation
  - Post-hoc probability calibration
  - Epistemic uncertainty quantification
  - Subgroup parity and distribution-shift auditing

---

## Image & Record Quality
- **Issue**: Medical images may contain blur, illumination variations, or artifacts; clinical tables contain missingness and noisy coding.
- **Existing Mitigations**:
  - Image quality assessment and outlier characterization
  - Missingness profiling and MCAR stress testing
  - Physiological bounds validation and schema enforcement

---

# Current Verification Status

## Completed
- Dataset integrity verification across all three modalities
- Metadata generation and canonical splitting
- Patient-group duplicate and leakage audits
- Independent modality baseline and architecture benchmarking
- Independent post-hoc explainability (Grad-CAM / TreeSHAP)
- Independent probability calibration (Temperature, Vector, Isotonic)
- Independent uncertainty estimation (MC Dropout, Bootstrap Ensembles)
- Clinical distribution-shift and robustness auditing
- Modality-level inference integration and end-to-end acceptance gates

## Next Research Stage: ACARA-U Multimodal Fusion
The next stage is to investigate whether the independently evaluated Retina, Foot Ulcer, and Clinical modalities provide complementary information when combined at the decision level.

The fusion study will evaluate:
- Individual-modality baselines
- Pairwise fusion
- Three-modality fusion
- Modality ablations
- Missing-modality scenarios
- Modality reliability and uncertainty weighting
- Fusion calibration
- Fusion robustness
- Final end-to-end evaluation

---

**Last Updated:** 2026-10-02  
**Project:** FusionMedAI  
**Document Version:** Dataset Verification v5
