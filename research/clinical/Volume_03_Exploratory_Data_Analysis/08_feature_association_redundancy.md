# Phase C3 — Exploratory Data Analysis: Feature Association & Redundancy (C3.8)

## 1. Objectives & Methodological Caution

This analysis examines pairwise correlations, categorical associations, and collinearity structures within the training population ($N = 69,519$).

> [!IMPORTANT]
> **Association $\ne$ Clinical Causation**: All statistical associations reported herein represent observational correlations in historical EHR data and must not be interpreted as causal determinants of hospital readmission.

---

## 2. Numerical Feature Correlation Matrix (Spearman Rank Correlation)

| Feature | `time_in_hosp` | `num_lab` | `num_proc` | `num_meds` | `num_outp` | `num_emerg` | `num_inpat` | `num_diag` | Target ($y=1$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `time_in_hospital` | $1.00$ | $+0.32$ | $+0.19$ | $+0.46$ | $-0.01$ | $-0.01$ | $+0.07$ | $+0.22$ | $+0.05$ |
| `num_lab_procedures` | $+0.32$ | $1.00$ | $+0.06$ | $+0.27$ | $-0.01$ | $-0.00$ | $+0.04$ | $+0.15$ | $+0.03$ |
| `num_procedures` | $+0.19$ | $+0.06$ | $1.00$ | $+0.39$ | $-0.02$ | $-0.04$ | $-0.07$ | $+0.07$ | $-0.01$ |
| `num_medications` | $+0.46$ | $+0.27$ | $+0.39$ | $1.00$ | $+0.05$ | $+0.01$ | $+0.06$ | $+0.26$ | $+0.04$ |
| `number_outpatient` | $-0.01$ | $-0.01$ | $-0.02$ | $+0.05$ | $1.00$ | $+0.09$ | $+0.11$ | $+0.09$ | $+0.03$ |
| `number_emergency` | $-0.01$ | $-0.00$ | $-0.04$ | $+0.01$ | $+0.09$ | $1.00$ | $+0.27$ | $+0.06$ | $+0.06$ |
| `number_inpatient` | $+0.07$ | $+0.04$ | $-0.07$ | $+0.06$ | $+0.11$ | $+0.27$ | $1.00$ | $+0.10$ | **$+0.12$** |
| `number_diagnoses` | $+0.22$ | $+0.15$ | $+0.07$ | $+0.26$ | $+0.09$ | $+0.06$ | $+0.10$ | $1.00$ | $+0.07$ |

---

## 3. Categorical Associations & Redundancy Register

1. **`diabetesMed` vs. Specific Medication Columns**:
   - `diabetesMed = 'No'` is an exact functional determinant for all 21 medication columns being `'No'`.
   - **Collinearity Handling**: Retaining both `diabetesMed` and 21 medication variables is well-handled by tree-based models, but for linear models, `diabetesMed` may create mild collinearity.
2. **`change` vs. Active Dosage Adjustments**:
   - `change = 'Ch'` has a high contingency association with whether any individual medication has `'Up'` or `'Down'` state (Cramér's $V \approx 0.72$).
3. **`time_in_hospital` and `num_medications`**:
   - Moderate positive correlation ($r = +0.46$), reflecting that longer hospital stays accumulate larger medication administration records.
4. **Prior Utilization Collinearity**:
   - `number_emergency` and `number_inpatient` exhibit moderate positive correlation ($r = +0.27$), capturing frequent healthcare utilization patterns.

> [!NOTE]
> **No Premature Feature Elimination**: Moderate correlations ($r < 0.80$) are retained without manual exclusion. Both variables in correlated pairs provide distinct clinical information and will be evaluated directly by baseline models in Phase C4.
