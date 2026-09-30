# Phase C2 — Clinical Data Pipeline: Patient-Grouped Canonical Split (C2.7)

## 1. Frozen Canonical Split Specification

The canonical split for the Clinical Modality is constructed from the **Eligible Readmission Cohort** ($99,343$ encounters from $69,990$ unique patients) using deterministic patient grouping.

### Partitioning Parameters:
- **Random Seed**: `42`
- **Target Proportions**: $70\%$ Training, $15\%$ Validation, $15\%$ Testing (at patient level)
- **Grouping Key**: `patient_nbr`
- **Stratification Anchor**: Patient-level early readmission history
- **Physical Output Directory**: `datasets/clinical/processed/splits/`

---

## 2. Quantitative Partition Statistics

| Split Partition | Unique Patients | % of Patients | Total Encounters | % of Encounters | Early Readmission (<30d) Count | Positive Target Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Training Set** (`train.csv`) | **$48,993$** | $70.00\%$ | **$69,519$** | $69.98\%$ | $7,916$ | **$11.39\%$** |
| **Validation Set** (`val.csv`) | **$10,498$** | $15.00\%$ | **$14,911$** | $15.01\%$ | $1,740$ | **$11.67\%$** |
| **Test Set** (`test.csv`) | **$10,499$** | $15.00\%$ | **$14,913$** | $15.01\%$ | $1,658$ | **$11.12\%$** |
| **Total Eligible Cohort** | **$69,990$** | **$100.00\%$** | **$99,343$** | **$100.00\%$** | **$11,314$** | **$11.39\%$** |

---

## 3. Physical Split Artifacts

The split artifacts are saved in immutable tabular formats under `datasets/clinical/processed/splits/`:

1. `train.csv` ($15,216,481$ bytes, $69,519$ rows): Complete feature table for model training.
2. `val.csv` ($3,231,865$ bytes, $14,911$ rows): Feature table for hyperparameter tuning and early stopping.
3. `test.csv` ($3,249,163$ bytes, $14,913$ rows): Held-out evaluation table for unbiased performance benchmarking.
4. `index.csv` ($3,041,984$ bytes, $99,343$ rows): Lightweight identifier manifest mapping each `encounter_id` and `patient_nbr` to its assigned `split` and target label.
5. `split_metadata.json` (Summary metadata record capturing all invariants and checksums).
