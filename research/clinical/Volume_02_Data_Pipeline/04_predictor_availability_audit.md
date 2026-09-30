# Phase C2 — Clinical Data Pipeline: Predictor Availability Audit (C2.4)

## 1. Decision Boundary: Discharge-Time Risk Stratification

The frozen clinical task is defined as **Discharge-Time Risk Stratification** (evaluating readmission risk at the point when discharge planning is finalized).

Every candidate predictor is formally audited against this boundary:

```
[Admission] ────────────── Inpatient Stay ──────────────► [Discharge Planning Boundary] ───► [Post-Discharge]
                                                          │
                                                          ├── Valid Predictors Available
                                                          └── Post-Discharge Info Unavailable
```

---

## 2. Predictor Availability Matrix

| Column Name | Category | Available at Discharge? | Audit Decision | Clinical / Methodological Rationale |
| :--- | :--- | :---: | :---: | :--- |
| `encounter_id` | Identifier | Yes (Admin) | **EXCLUDE** | Database sequence artifact; risks chronological memorization. |
| `patient_nbr` | Identifier | Yes (Admin) | **EXCLUDE** | High-cardinality identity field; used strictly as grouping constraint. |
| `race` | Demographic | Yes | **INCLUDE** | Recorded at admission/intake. |
| `gender` | Demographic | Yes | **INCLUDE** | Recorded at admission/intake. |
| `age` | Demographic | Yes | **INCLUDE** | Recorded at admission/intake. |
| `weight` | Demographic | Yes | **FLAGGED** | 96.86% missing (`'?'`); uninformative in raw state. |
| `admission_type_id` | Encounter Context | Yes | **INCLUDE** | Fixed at intake. |
| `admission_source_id` | Encounter Context | Yes | **INCLUDE** | Fixed at intake. |
| `medical_specialty` | Encounter Context | Yes | **INCLUDE** | Admitting/attending physician specialty. |
| `time_in_hospital` | Inpatient Metric | Yes | **INCLUDE** | Completed stay duration known at discharge planning. |
| `num_lab_procedures` | Inpatient Metric | Yes | **INCLUDE** | Cumulative lab tests ordered during stay known at discharge. |
| `num_procedures` | Inpatient Metric | Yes | **INCLUDE** | Total procedures performed during stay known at discharge. |
| `num_medications` | Inpatient Metric | Yes | **INCLUDE** | Distinct medications administered during stay known at discharge. |
| `number_outpatient` | Prior Utilization | Yes | **INCLUDE** | Prior 12-month historical outpatient visits. |
| `number_emergency` | Prior Utilization | Yes | **INCLUDE** | Prior 12-month historical emergency visits. |
| `number_inpatient` | Prior Utilization | Yes | **INCLUDE** | Prior 12-month historical inpatient admissions. |
| `diag_1`, `diag_2`, `diag_3` | Diagnosis | Yes | **INCLUDE** | Finalized billing/clinical diagnoses assigned for the encounter. |
| `number_diagnoses` | Diagnosis | Yes | **INCLUDE** | Total diagnosis codes documented for the stay. |
| `max_glu_serum` | Glycemic Lab | Yes | **INCLUDE** | In-hospital lab result; `'None'` retained as valid category. |
| `A1Cresult` | Glycemic Lab | Yes | **INCLUDE** | In-hospital lab result; `'None'` retained as valid category. |
| 23 Medication Features | Pharmacotherapy | Yes | **INCLUDE** | Final inpatient medication regimens and titration status. |
| `examide`, `citoglipton` | Medication | Yes | **EXCLUDE** | Zero-variance (100% `'No'`); zero predictive information. |
| `change` | Treatment Dynamic | Yes | **INCLUDE** | In-hospital medication changes finalized during stay. |
| `diabetesMed` | Treatment Dynamic | Yes | **INCLUDE** | In-hospital active diabetic medication indicator. |
| `payer_code` | Administrative | Yes | **FLAGGED** | 39.56% missing; non-clinical billing tag. |
| `discharge_disposition_id` | Cohort / Context | Yes | **COHORT FILTER** | Used to identify expired/hospice cases for cohort eligibility. |
| `readmitted` | Target | Post-Discharge | **TARGET FIELD** | Dependent outcome ($y \in \{0, 1\}$); strictly isolated. |

---

## 3. Summary of Predictor Audit

- **Predictive Candidate Attributes**: 45 valid attributes (excluding 2 identifiers, 2 zero-variance medications, and target).
- **Zero Lookahead Leakage**: Because prediction is executed at discharge planning, all completed-stay attributes are legitimate.
