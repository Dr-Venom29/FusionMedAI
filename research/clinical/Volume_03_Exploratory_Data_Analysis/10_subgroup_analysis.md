# Phase C3 — Exploratory Data Analysis: Clinical Subgroup Analysis (C3.10)

## 1. Scope & Analytical Context

This exploratory subgroup analysis audits 30-day early readmission rates ($y=1$) across demographic, clinical, and utilization strata within the training cohort ($N = 69,519$).

> [!NOTE]
> **Exploratory Characterization**: Subgroup variations reported here serve to uncover clinical risk gradients and potential confounding structures. Formal algorithmic fairness audits will be conducted post-modeling in subsequent phases.

---

## 2. Demographic Subgroup Readmission Rates

| Demographic Dimension | Subgroup Category | Training Encounters | % of Cohort | 30-Day Readmission Count | Early Readmission Rate ($y=1$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Age Band** | `[0-10)` | $105$ | $0.15\%$ | $4$ | $3.81\%$ |
| | `[10-20)` | $462$ | $0.66\%$ | $45$ | $9.74\%$ |
| | `[20-30)` | $1,114$ | $1.60\%$ | $132$ | $11.85\%$ |
| | `[30-40)` | $2,605$ | $3.75\%$ | $302$ | $11.59\%$ |
| | `[40-50)` | $6,664$ | $9.59\%$ | $741$ | $11.12\%$ |
| | `[50-60)` | $11,922$ | $17.15\%$ | $1,348$ | $11.31\%$ |
| | `[60-70)` | $15,310$ | $22.02\%$ | $1,714$ | $11.20\%$ |
| | **`[70-80)`** | **$17,794$** | **$25.59\%$** | **$2,192$** | **$12.32\%$** |
| | `[80-90)` | $11,634$ | $16.74\%$ | $1,288$ | $11.07\%$ |
| | `[90-100)` | $1,909$ | $2.75\%$ | $150$ | $7.86\%$ |
| **Race** | Caucasian | $52,001$ | $74.80\%$ | $5,966$ | $11.47\%$ |
| | AfricanAmerican | $13,107$ | $18.85\%$ | $1,528$ | $11.66\%$ |
| | Hispanic | $1,399$ | $2.01\%$ | $145$ | $10.36\%$ |
| | Asian | $436$ | $0.63\%$ | $39$ | $8.94\%$ |
| | Other | $1,032$ | $1.48\%$ | $104$ | $10.08\%$ |
| | `?` (Unknown) | $1,544$ | $2.22\%$ | $134$ | $8.68\%$ |
| **Gender** | Female | $37,344$ | $53.72\%$ | $4,284$ | $11.47\%$ |
| | Male | $32,173$ | $46.28\%$ | $3,632$ | $11.29\%$ |

---

## 3. Clinical & Pharmacotherapy Subgroups

| Clinical Dimension | Subgroup Category | Training Encounters | % of Cohort | Early Readmission Rate ($y=1$) | Risk Differential |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Medication Change** | `Ch` (Changed) | $32,178$ | $46.29\%$ | **$12.35\%$** | $+1.80\%$ higher readmission rate |
| | `No` (Unchanged) | $37,341$ | $53.71\%$ | **$10.55\%$** | Baseline stability |
| **Diabetes Medication**| `Yes` (Prescribed) | $53,522$ | $76.99\%$ | **$12.01\%$** | Pharmacotherapy cohort |
| | `No` (Unprescribed)| $15,997$ | $23.01\%$ | **$9.30\%$** | Diet-managed / unmedicated stay |
| **Insulin Status** | `Down` (Decreased) | $8,294$ | $11.93\%$ | **$13.91\%$** | Highest medication risk group |
| | `Up` (Increased) | $7,725$ | $11.11\%$ | **$13.57\%$** | Unstable glycemic titration |
| | `Steady` (Maintained)| $21,123$ | $30.38\%$ | **$11.75\%$** | Stable maintenance |
| | `No` (Not Prescribed)| $32,377$ | $46.57\%$ | **$10.02\%$** | Lowest insulin risk |
| **Serum Glucose Test** | `>300` | $875$ | $1.26\%$ | **$14.86\%$** | Acute severe hyperglycemia |
| | `>200` | $1,006$ | $1.45\%$ | **$13.42\%$** | Moderate acute hyperglycemia |
| | `'None'` | $65,868$ | $94.75\%$ | **$11.33\%$** | Test not ordered |
| | `'Norm'` | $1,770$ | $2.55\%$ | **$10.51\%$** | Controlled acute state |

---

## 4. Key Subgroup Insights

1. **Age-Stratified Risk Profile**:
   - Readmission risk increases from pediatric admissions ($3.81\%$) to peak in the septuagenarian group `[70-80)` at **$12.32\%$**.
   - The `[90-100)` group has a lower observed readmission rate ($7.86\%$). The underlying reason for this pattern is not established by this analysis.
2. **Insulin Instability as Acute Flag**:
   - Encounters where insulin was adjusted (`Up` or `Down`) experience nearly a **$40\%$ relative increase in early readmission risk** compared to non-insulin patients.
3. **Observed Baseline Demographic Rates**:
   - Caucasian: $11.47\%$ ($5,966 / 52,001$)
   - African American: $11.66\%$ ($1,528 / 13,107$)
   - These crude rates are similar within this cohort; this is not a formal fairness assessment. Formal algorithmic fairness audits will be evaluated in subsequent modeling stages.
