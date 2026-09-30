# Phase C3 — Exploratory Data Analysis: Feature Representation Decisions & Contract (C3.11)

## 1. Frozen Feature Representation Contract

This document codifies the definitive transformation and encoding decisions for every attribute in the Clinical Modality. Baseline models in Phase C4 and benchmark architectures in Phase C5 must strictly execute this contract.

```mermaid
flowchart TD
    subgraph Feature_Representation_Contract [Clinical Representation Contract]
        ID["Identifiers (2)<br/>encounter_id, patient_nbr"] -->|Strictly Omit| DROP1[DROPPED]
        ZV["Zero-Variance (2)<br/>examide, citoglipton"] -->|Zero Variance| DROP2[DROPPED]
        
        NUM["Numerical Features (8)<br/>time_in_hospital, num_medications, etc."] -->|Fitted on Train Only| SCALE[Standard / Robust Scaling]
        DEM["Demographics (3)<br/>race, gender, age"] -->|One-Hot / Ordinal| CAT1[Encoded Categoricals]
        CTX["Context & Admin (3)<br/>admission_type/source, specialty"] -->|Top-K / Frequency| CAT2[Encoded Context]
        LAB["Glycemic Labs (2)<br/>max_glu_serum, A1Cresult"] -->|Preserve 'None'| CAT3[4-State Categorical]
        DIAG["ICD-9 Diagnoses (3)<br/>diag_1, diag_2, diag_3"] -->|9 Clinical Chapters| CAT4[Clinical Chapter One-Hot]
        MED["Diabetic Medications (21)<br/>insulin, metformin, sulfonylureas..."] -->|Exposure States| CAT5[4-Level Ordinal (0,1,2,3)]
        DYN["Treatment Dynamics (2)<br/>change, diabetesMed"] -->|Binary Flags| CAT6[Binary (0/1)]
    end
```

---

## 2. Complete Attribute-by-Attribute Representation Matrix

| Raw Attribute | Physical Dtype | Preprocessing Action | Missingness / Semantic Rule | Final Model Feature Representation |
| :--- | :--- | :--- | :--- | :--- |
| `encounter_id` | `int64` | **DROP** | Preserved only in `index.csv` manifest. | *Excluded from feature matrix* |
| `patient_nbr` | `int64` | **DROP** | Preserved only for group-splitting. | *Excluded from feature matrix* |
| `examide` | `object` | **DROP** | Constant (100% `'No'`). | *Excluded from feature matrix* |
| `citoglipton` | `object` | **DROP** | Constant (100% `'No'`). | *Excluded from feature matrix* |
| `weight` | `object` | **DROP** | $96.88\%$ missing; high sparsity. | *Excluded from feature matrix* |
| `time_in_hospital` | `int64` | **SCALE** | Complete. Scale fitted on train. | Continuous Scaled Float |
| `num_lab_procedures`| `int64` | **SCALE** | Complete. Scale fitted on train. | Continuous Scaled Float |
| `num_procedures` | `int64` | **SCALE** | Complete. Scale fitted on train. | Continuous Scaled Float |
| `num_medications` | `int64` | **SCALE** | Complete. Scale fitted on train. | Continuous Scaled Float |
| `number_outpatient` | `int64` | **SCALE / LOG1P**| Zero-inflated count. | Continuous Scaled Float |
| `number_emergency` | `int64` | **SCALE / LOG1P**| Zero-inflated count. | Continuous Scaled Float |
| `number_inpatient` | `int64` | **SCALE / LOG1P**| Zero-inflated count. | Continuous Scaled Float |
| `number_diagnoses` | `int64` | **SCALE** | Complete. Scale fitted on train. | Continuous Scaled Float |
| `race` | `object` | **ONE-HOT** | Map `'?'` to `"Unknown"`. | 6-Dim Binary Vector |
| `gender` | `object` | **ONE-HOT** | Map `"Unknown/Invalid"` to mode. | 2-Dim Binary Vector |
| `age` | `object` | **ORDINAL** | Map `[0-10)` to $0$, ..., `[90-100)` to $9$. | 1-Dim Ordinal Integer ($0–9$) |
| `admission_type_id` | `int64` | **TOP-K / ONE-HOT**| Map 5, 6, 8 to `"Missing/Other"`. | 5-Dim Binary Vector |
| `admission_source_id`| `int64`| **TOP-K / ONE-HOT**| Map 9, 17, 20 to `"Missing/Other"`. | 6-Dim Binary Vector |
| `medical_specialty` | `object` | **TOP-10 + OTHER**| Top 10 specialties, other, unknown (`'?'`). | 12-Dim Binary Vector |
| `payer_code` | `object` | **TOP-8 + OTHER** | Top 8 commercial/govt, other, unknown (`'?'`). | 10-Dim Binary Vector |
| `max_glu_serum` | `object` | **4-STATE ORDINAL**| Map `'None'`: $0$, `'Norm'`: $1$, `'>200'`: $2$, `'>300'`: $3$. | 1-Dim Ordinal Integer ($0–3$) |
| `A1Cresult` | `object` | **4-STATE ORDINAL**| Map `'None'`: $0$, `'Norm'`: $1$, `'>7'`: $2$, `'>8'`: $3$. | 1-Dim Ordinal Integer ($0–3$) |
| `diag_1` | `object` | **ICD-9 CHAPTER** | Map to 9 clinical disease chapters. | 9-Dim Binary Vector |
| `diag_2` | `object` | **ICD-9 CHAPTER** | Map to 9 clinical disease chapters. | 9-Dim Binary Vector |
| `diag_3` | `object` | **ICD-9 CHAPTER** | Map to 9 clinical disease chapters. | 9-Dim Binary Vector |
| 21 Active Meds | `object` | **4-LEVEL ORDINAL**| Map `'No'`: $0$, `'Steady'`: $1$, `'Up'`: $2$, `'Down'`: $3$. | $21 \times 1$-Dim Ordinal Ints ($0–3$) |
| `change` | `object` | **BINARY** | Map `'No'`: $0$, `'Ch'`: $1$. | 1-Dim Binary Indicator ($0/1$) |
| `diabetesMed` | `object` | **BINARY** | Map `'No'`: $0$, `'Yes'`: $1$. | 1-Dim Binary Indicator ($0/1$) |
| `readmitted` | `object` | **TARGET** | Map `'<30'`: $1$, `'>30'`: $0$, `'NO'`: $0$. | **Dependent Target ($y \in \{0, 1\}$)** |

---

## 3. Strict Preprocessing Isolation Rules

1. **Estimator Fitting**: All transformations (imputers, one-hot encoders, scalers, target encoders) must be **fitted strictly on the training partition (`train.csv`)**.
2. **Validation/Test Isolation**: The validation and test sets must be transformed using the exact pre-fitted training transformer pipeline without parameter re-estimation.
3. **Reproducibility Manifest**: This representation contract is exported to `datasets/clinical/metadata/eda/reports/feature_representation_contract.json`.
