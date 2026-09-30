# Phase C2 — Clinical Data Pipeline: Pipeline Contract & Preprocessing Boundaries (C2.9)

## 1. Pipeline Execution Flow

Phase C2 establishes the immutable data transformation and isolation contract connecting Phase C1 to subsequent analytical and modeling phases:

```
C1 Frozen Raw Dataset (101,766 Encounters / 71,518 Patients)
        │
        ▼
C2 Eligible Readmission Cohort (99,343 Encounters / 69,990 Patients)
   [2,423 Expired/Hospice Cases Logged to interim/cohort_eligibility.csv]
        │
        ▼
C2 Canonical Patient-Grouped Stratified Split (Seed 42)
   ├── TRAIN: 69,519 Encounters (48,993 Patients) [70.0%]
   ├── VAL:   14,911 Encounters (10,498 Patients) [15.0%]
   └── TEST:  14,913 Encounters (10,499 Patients) [15.0%]
        │
        ▼
Frozen Split Manifests (datasets/clinical/processed/splits/)
        │
        ▼
Phase C3: Exploratory Data Analysis & Feature Representation Fitting
```

---

## 2. Pre-Fitting Boundary Rules for Phase C3+

To prevent data snooping and information leakage into evaluation sets:

1. **Training-Only Estimator Fitting**:
   - Any numerical standardizers, min-max scalers, target encoders, or imputation statistics must be **fitted strictly on `train.csv`**.
   - The validation (`val.csv`) and test (`test.csv`) sets must only be transformed using parameters learned on `train.csv`.
2. **Missingness Preservation**:
   - `'None'` in `max_glu_serum` and `A1Cresult` remains a discrete categorical state ("Not Measured / Not Recorded").
   - Imputation must not be applied to lab test absence indicators.
3. **Identifier Isolation**:
   - `encounter_id` and `patient_nbr` must remain isolated in index manifests and never enter feature tensors or model input matrices.
4. **Zero-Variance Removal**:
   - `examide` and `citoglipton` must be dropped during pipeline preprocessing as verified in Phase C1.

---

## 3. Phase C2 Contract Sign-Off

With the successful execution and verification of Phase C2:
- The clinical cohort is rigorously defined.
- Patient-grouped split isolation is deterministically frozen.
- The pipeline contract is established for Phase C3.
