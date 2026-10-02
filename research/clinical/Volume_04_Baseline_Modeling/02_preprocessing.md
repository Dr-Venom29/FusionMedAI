# Phase C4 — Baseline Modeling: Feature Preprocessing & Transformation Pipeline (C4.2)

## 1. Preprocessing Pipeline Architecture

The feature transformation pipeline is implemented in [`src/clinical/modeling/preprocessing.py`](../../../src/clinical/modeling/preprocessing.py). It strictly operationalizes the frozen Phase C3 representation contract:

```mermaid
flowchart TD
    RAW["Raw Inpatient Record (47 Attributes)"] --> SPLIT["Domain Separation"]
    
    SPLIT --> NUM["Numerical Features (8)<br/>time_in_hospital, num_medications, etc."]
    SPLIT --> DEM["Demographics (3)<br/>age, gender, race"]
    SPLIT --> CTX["Encounter Context (4)<br/>admission_type/source, specialty, payer"]
    SPLIT --> LAB["Glycemic Labs (2)<br/>max_glu_serum, A1Cresult"]
    SPLIT --> DIAG["ICD-9 Diagnoses (3)<br/>diag_1, diag_2, diag_3"]
    SPLIT --> MED["Diabetic Medications (21)<br/>insulin, metformin, etc."]
    SPLIT --> DYN["Treatment Dynamics (2)<br/>change, diabetesMed"]

    NUM --> SCALE["StandardScaler<br/>(Fitted on Train)"]
    DEM --> DEM_ENC["OneHotEncoder (Race, Gender)<br/>Ordinal (Age [0-9])"]
    CTX --> CTX_ENC["OneHotEncoder (Type, Source)<br/>Top-10 Specialty + Top-8 Payer"]
    LAB --> LAB_ENC["4-State Ordinal (0, 1, 2, 3)<br/>'None' Preserved as 0"]
    DIAG --> DIAG_ENC["9-Chapter ICD-9 Mapping<br/>OneHotEncoder across 3 fields"]
    MED --> MED_ENC["4-Level Exposure (0, 1, 2, 3)<br/>No=0, Steady=1, Up=2, Down=3"]
    DYN --> DYN_ENC["Binary Encoding (0 / 1)"]

    SCALE & DEM_ENC & CTX_ENC & LAB_ENC & DIAG_ENC & MED_ENC & DYN_ENC --> CONCAT["Unified Feature Matrix X<br/>(D = 119 Columns)"]
```

---

## 2. Feature Dimension Accounting ($D = 119$)

| Feature Domain | Raw Features | Transformed Dimension | Encoding & Transformation Details |
| :--- | :---: | :---: | :--- |
| **Numerical Intensity & Utilization** | $8$ | $8$ | Standardized continuous scaling ($\mu=0, \sigma=1$ fitted on train). |
| **Demographics** | $3$ | $9$ | Age ($1$ ordinal) + Race ($6$ OHE) + Gender ($2$ OHE). |
| **Encounter Context & Admin** | $4$ | $42$ | Admission type ($8$ OHE) + Source ($17$ OHE) + Top-10 Specialty ($12$ OHE) + Top-8 Payer ($10$ OHE - reduced to non-empty). |
| **Glycemic Monitoring Labs** | $2$ | $2$ | 4-State Ordinal integers ($0, 1, 2, 3$ preserving `'None'` as baseline state $0$). |
| **ICD-9 Clinical Diagnoses** | $3$ | $33$ | 9 Disease chapters + External + Missing ($11$ categories $\times 3$ diagnosis fields). |
| **Diabetic Pharmacotherapy** | $21$ | $21$ | 4-Level exposure states ($0=\text{No}, 1=\text{Steady}, 2=\text{Up}, 3=\text{Down}$). |
| **Treatment Dynamics** | $2$ | $2$ | Binary indicators ($change \in \{0, 1\}$, $diabetesMed \in \{0, 1\}$). |
| **Total Feature Matrix** | **$43$ Active Inputs** | **$119$ Dimensions** | Complete numerical array $X \in \mathbb{R}^{N \times 119}$. |

---

## 3. Strict Preprocessing Invariants

1. **Estimator Isolation**:
   - `StandardScaler`, `OneHotEncoder`, `top_specialties_`, and `top_payers_` are fitted exclusively during `preprocessor.fit(df_train)`.
   - `preprocessor.transform(df_val)` and `preprocessor.transform(df_test)` apply these fixed parameter matrices without updating internal states.
2. **Handling Unseen Categories**:
   - All `OneHotEncoder` instances are configured with `handle_unknown="ignore"`, preventing runtime exceptions on novel levels.
3. **Traceability Logging**:
   - Patient numbers and encounter IDs are separated into isolated trace DataFrames, guaranteeing zero identifier contamination in $X$.
