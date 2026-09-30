# Phase C1 — Dataset & Clinical Task Audit: Schema & Column Structure

## 1. Schema Overview

The raw file `diabetic_data.csv` comprises **101,766 encounter rows** across **50 attributes**. Every column has been audited for its physical data type, cardinality (number of distinct values), missingness representation, and unexpected values.

---

## 2. Complete 50-Column Attribute Inventory

| # | Column Name | Raw Dtype | Cardinality | Missing Repr. | Distinct Examples / Range | Description ("What is this column?") |
| :- | :--- | :--- | :- | :--- | :--- | :--- |
| 1 | `encounter_id` | `int64` | 101,766 | None | 12570 to 443867222 | Unique numerical identifier for each hospital encounter |
| 2 | `patient_nbr` | `int64` | 71,518 | None | 135 to 189502619 | Unique numerical identifier for each individual patient |
| 3 | `race` | `object` | 6 | `'?'` (2,273) | Caucasian, AfricanAmerican, ?, Hispanic, Other, Asian | Reported racial demographic category of the patient |
| 4 | `gender` | `object` | 3 | Invalid (3) | Female (54,708), Male (47,055), Unknown/Invalid (3) | Administrative gender of the patient |
| 5 | `age` | `object` | 10 | None | `[0-10)`, `[10-20)`, ..., `[90-100)` | 10-year age band at encounter admission |
| 6 | `weight` | `object` | 10 | `'?'` (98,569) | `?`, `[0-25)`, `[25-50)`, `[50-75)`, `[75-100)`, ... | Weight in 25-pound intervals; 96.86% unrecorded (`'?'`) |
| 7 | `admission_type_id` | `int64` | 8 | ID codes | 1 (Emergency), 2 (Urgent), 3 (Elective), 4, 5, 6, 7, 8 | Type of admission into the hospital |
| 8 | `discharge_disposition_id` | `int64` | 26 | ID codes | 1 (Home), 2, 3 (SNF), 6 (Home Health), 11 (Expired), ... | Patient disposition at hospital discharge |
| 9 | `admission_source_id` | `int64` | 17 | ID codes | 1 (Physician Ref), 7 (ER), 2 (Clinic Ref), 4, ... | Source/referral mechanism leading to admission |
| 10 | `time_in_hospital` | `int64` | 14 | None | 1 to 14 days (Mean: 4.40, Median: 4.0) | Total duration of the inpatient stay in whole days |
| 11 | `payer_code` | `object` | 18 | `'?'` (40,256) | `?`, `MC`, `MD`, `HM`, `UN`, `BC`, `SP`, `CP`, ... | Payer / health insurer code (39.56% `'?'`) |
| 12 | `medical_specialty` | `object` | 73 | `'?'` (49,949) | `?`, InternalMedicine, Emergency/Trauma, Cardiology, ... | Specialty of the admitting physician (49.08% `'?'`) |
| 13 | `num_lab_procedures` | `int64` | 118 | None | 1 to 132 (Mean: 43.10, Median: 44.0) | Count of lab tests performed during encounter |
| 14 | `num_procedures` | `int64` | 7 | None | 0 to 6 (Mean: 1.34, Median: 1.0) | Count of non-lab procedures performed during stay |
| 15 | `num_medications` | `int64` | 75 | None | 1 to 81 (Mean: 16.02, Median: 15.0) | Total distinct medications administered during stay |
| 16 | `number_outpatient` | `int64` | 39 | None | 0 to 42 (Mean: 0.37, Median: 0.0) | Outpatient visits in the 12 months prior to encounter |
| 17 | `number_emergency` | `int64` | 33 | None | 0 to 76 (Mean: 0.20, Median: 0.0) | Emergency visits in the 12 months prior to encounter |
| 18 | `number_inpatient` | `int64` | 21 | None | 0 to 21 (Mean: 0.64, Median: 0.0) | Inpatient admissions in the 12 months prior to encounter |
| 19 | `diag_1` | `object` | 717 | `'?'` (21) | ICD-9 codes: 250.83, 276, 428, 414, 410, 786, ... | Primary diagnosis code for the hospital stay |
| 20 | `diag_2` | `object` | 749 | `'?'` (358) | ICD-9 codes: 250.01, 276, 428, 401, 414, 250, ... | Secondary diagnosis code |
| 21 | `diag_3` | `object` | 790 | `'?'` (1,423) | ICD-9 codes: 250, 401, 272, 428, 250.02, ... | Additional/tertiary diagnosis code |
| 22 | `number_diagnoses` | `int64` | 16 | None | 1 to 16 (Mean: 7.42, Median: 8.0) | Total diagnosis codes entered for the encounter |
| 23 | `max_glu_serum` | `object` | 4 (3 + None) | `'None'` (96,420) | `None` (94.75%), `Norm` (2.55%), `>200` (1.46%), `>300` (1.24%) | Result of serum glucose test during encounter |
| 24 | `A1Cresult` | `object` | 4 (3 + None) | `'None'` (84,748) | `None` (83.28%), `>8` (8.07%), `Norm` (4.90%), `>7` (3.75%) | Result of HbA1c lab test during encounter |
| 25 | `metformin` | `object` | 4 | None | No, Steady, Up, Down | Metformin administration & dosage adjustment |
| 26 | `repaglinide` | `object` | 4 | None | No, Steady, Up, Down | Repaglinide administration & dosage adjustment |
| 27 | `nateglinide` | `object` | 4 | None | No, Steady, Up, Down | Nateglinide administration & dosage adjustment |
| 28 | `chlorpropamide` | `object` | 4 | None | No, Steady, Up, Down | Chlorpropamide administration & dosage adjustment |
| 29 | `glimepiride` | `object` | 4 | None | No, Steady, Up, Down | Glimepiride administration & dosage adjustment |
| 30 | `acetohexamide` | `object` | 2 | None | No (101,765), Steady (1) | Acetohexamide administration & dosage adjustment |
| 31 | `glipizide` | `object` | 4 | None | No, Steady, Up, Down | Glipizide administration & dosage adjustment |
| 32 | `glyburide` | `object` | 4 | None | No, Steady, Up, Down | Glyburide administration & dosage adjustment |
| 33 | `tolbutamide` | `object` | 2 | None | No (101,743), Steady (23) | Tolbutamide administration & dosage adjustment |
| 34 | `pioglitazone` | `object` | 4 | None | No, Steady, Up, Down | Pioglitazone administration & dosage adjustment |
| 35 | `rosiglitazone` | `object` | 4 | None | No, Steady, Up, Down | Rosiglitazone administration & dosage adjustment |
| 36 | `acarbose` | `object` | 4 | None | No, Steady, Up, Down | Acarbose administration & dosage adjustment |
| 37 | `miglitol` | `object` | 4 | None | No, Steady, Up, Down | Miglitol administration & dosage adjustment |
| 38 | `troglitazone` | `object` | 2 | None | No (101,763), Steady (3) | Troglitazone administration & dosage adjustment |
| 39 | `tolazamide` | `object` | 3 | None | No (101,727), Steady (38), Up (1) | Tolazamide administration & dosage adjustment |
| 40 | `examide` | `object` | 1 | None | No (101,766) — **Zero Variance** | Examide administration (all 'No') |
| 41 | `citoglipton` | `object` | 1 | None | No (101,766) — **Zero Variance** | Citoglipton administration (all 'No') |
| 42 | `insulin` | `object` | 4 | None | No (47,383), Steady (30,849), Down (12,218), Up (11,316) | Insulin administration & dosage adjustment |
| 43 | `glyburide-metformin` | `object` | 4 | None | No, Steady, Up, Down | Combination oral medication |
| 44 | `glipizide-metformin` | `object` | 2 | None | No (101,753), Steady (13) | Combination oral medication |
| 45 | `glimepiride-pioglitazone`| `object` | 2 | None | No (101,765), Steady (1) | Combination oral medication |
| 46 | `metformin-rosiglitazone` | `object` | 2 | None | No (101,764), Steady (2) | Combination oral medication |
| 47 | `metformin-pioglitazone` | `object` | 2 | None | No (101,765), Steady (1) | Combination oral medication |
| 48 | `change` | `object` | 2 | None | No (54,755), Ch (47,011) | Change in any diabetic medication dosage or prescription |
| 49 | `diabetesMed` | `object` | 2 | None | Yes (78,363), No (23,403) | Any diabetic medication prescribed or administered |
| 50 | `readmitted` | `object` | 3 | None | NO (54,864), >30 (35,545), <30 (11,357) | **Hospital Readmission Target** |

---

## 3. Zero-Variance & Degenerate Columns

Two medication columns contain **exactly one constant value across all 101,766 rows**:
- `examide`: 100.0% `'No'` (101,766 encounters)
- `citoglipton`: 100.0% `'No'` (101,766 encounters)

**Audit Finding**: Neither `examide` nor `citoglipton` provides any discriminative variance or clinical information. They must be logged for removal during pipeline construction in Phase C2, but remain intact in the frozen raw dataset.
