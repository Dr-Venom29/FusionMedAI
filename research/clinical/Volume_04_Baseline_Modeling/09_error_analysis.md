# Phase C4 — Baseline Modeling: Stratified Subgroup Error Analysis (C4.9)

## 1. Scope & Error Analysis Protocol

Error analysis examines where baseline models succeed and fail across distinct patient subgroups. Rather than relying on aggregate metrics alone, we inspect:
- **False Positives (FP)**: Patients predicted as high risk who did not experience early readmission (risk of unnecessary intervention / resource burden).
- **False Negatives (FN)**: Patients predicted as low risk who were subsequently readmitted within 30 days (clinical risk of missed intervention).

The analysis is conducted at the clinical operating threshold $\theta = 0.20$ across:
1. **Age Strata**: `[0-10)` through `[90-100)`
2. **Race Categories**: `Caucasian`, `AfricanAmerican`, `Hispanic`, `Asian`, `Other`, `Unknown`
3. **Gender**: `Female`, `Male`
4. **Primary Admitting Diagnosis Chapter**: `Circulatory`, `Respiratory`, `Diabetes`, `Digestive`, etc.
5. **Prior Inpatient Utilization History**: $0, 1, 2, \ge 3$ prior stays
6. **Insulin Exposure State**: `No`, `Steady`, `Up`, `Down`

---

## 2. Key Observational Patterns

1. **Prior Utilization Subgroup Variation**:
   - Prior inpatient utilization is an important predictive feature in the baseline models, with subgroup sensitivity varying substantially across utilization strata (e.g. higher sensitivity observed among patients with $\ge 2$ prior inpatient stays, where baseline risk is elevated).
2. **Diagnostic Chapter Heterogeneity**:
   - Encounters admitted for acute Circulatory and Respiratory conditions account for the largest absolute volume of both true positives and false positives due to their high cohort prevalence.
3. **Glycemic Instability Signal**:
   - Encounters with active insulin dosage adjustments (`Up` or `Down`) yield higher true positive rates than stable or unmedicated admissions.
4. **Observational Framing**:
   - All subgroup variations reported are purely empirical performance characterizations. No causal inferences are asserted regarding healthcare disparity mechanisms.
