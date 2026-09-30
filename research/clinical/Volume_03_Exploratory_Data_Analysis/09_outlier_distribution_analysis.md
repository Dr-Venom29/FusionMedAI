# Phase C3 — Exploratory Data Analysis: Outlier & Distribution Analysis (C3.9)

## 1. Outlier Philosophy in FusionMedAI

In medical AI research, unusual clinical observations often represent genuine high-acuity, complex multi-morbid cases rather than corrupted data.

> [!IMPORTANT]
> **Clinical Outlier Rule**: Statistical outliers must **not** be automatically deleted unless verified as physical impossibilities or catastrophic recording corruptions. Legitimate clinical variation must be preserved to evaluate model robustness on severe disease states.

---

## 2. Quantitative Extreme Value Audit (Training Cohort)

| Feature | 99th Percentile | Observed Maximum | Extreme Cases ($> 99\text{th}\%$) | Clinical Evaluation & Plausibility | Decision |
| :--- | :---: | :---: | :---: | :--- | :---: |
| `number_emergency` | $4.0$ visits | **$76$ visits** | $437$ encounters | Patients with severe chronic instability or substance use frequently utilize ER services. Plausible historical utilization. | **RETAIN** |
| `number_outpatient` | $5.0$ visits | **$42$ visits** | $524$ encounters | High outpatient volume reflects active specialized multi-clinic management (e.g., dialysis, wound care). Plausible. | **RETAIN** |
| `number_inpatient` | $5.0$ visits | **$21$ visits** | $643$ encounters | Frequent hospitalization cohort ("super-utilizers"). Core target population. | **RETAIN** |
| `num_medications` | $42.0$ meds | **$81$ meds** | $628$ encounters | Extreme polypharmacy in ICU/cardiac surgical patients. Clinically plausible. | **RETAIN** |
| `num_lab_procedures` | $89.0$ tests | **$132$ tests** | $671$ encounters | High-intensity ICU/stepdown monitoring across multi-day stays. Plausible. | **RETAIN** |
| `time_in_hospital` | $14.0$ days | **$14$ days** | $1,042$ encounters | Dataset inclusion rule strictly capped stays at $14$ days. Valid. | **RETAIN** |
| `number_diagnoses` | $9.0$ codes | **$16$ codes** | $1,289$ encounters | Multi-morbid diabetic patients with extensive secondary complications. Valid. | **RETAIN** |

### Standardized Feature Maximum Reference

| Feature | Observed Maximum in Training Set |
| :--- | :---: |
| `number_emergency` | **$76$** |
| `number_outpatient` | **$42$** |
| `number_inpatient` | **$21$** |
| `num_medications` | **$81$** |
| `num_lab_procedures` | **$132$** |
| `number_diagnoses` | **$16$** |
| `time_in_hospital` | **$14$** |

---

## 3. Categorical Rare States Audit

- `gender = 'Unknown/Invalid'`: Exactly **$3$ encounters** in the raw dataset ($2$ in train, $1$ in val).
  - *Decision*: Map to the dominant category or group into a designated "Unknown" level; do not discard.
- Rare combination pills (e.g., `acetohexamide`, `glimepiride-pioglitazone`, `metformin-rosiglitazone` with $\le 2$ cases):
  - *Decision*: Retain ordinal structure ($0/1$); their near-zero variance will naturally receive near-zero feature importance in tree models.
- Zero-variance features (`examide`, `citoglipton`):
  - *Decision*: Excluded from modeling as established in Phase C1.

---

## 4. Summary of Decisions

**Zero sample deletions** were performed as a result of the outlier audit. All extreme count values represent valid clinical phenomena and are retained in the training, validation, and test datasets.
