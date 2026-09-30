# Phase C1 — Dataset & Clinical Task Audit: Clinical & Epidemiological Limitations

## 1. Context for Peer-Reviewed Publication

In publishing research utilizing the UCI Diabetes 130-US Hospitals dataset, rigorous scientific integrity requires explicit disclosure of structural, temporal, and clinical boundaries. The following limitations must be carried into the final research papers and methodology sections of FusionMedAI.

---

## 2. Catalog of Formal Limitations

### 1. Retrospective EHR Observational Design (1999–2008)
- The dataset captures inpatient care delivered between 1999 and 2008 across 130 US hospital facilities.
- **Clinical Implication**: Clinical guidelines for diabetes management have evolved significantly since 2008 (e.g., emergence of SGLT-2 inhibitors, GLP-1 receptor agonists, continuous glucose monitors [CGMs], and modernized HbA1c targets). The dataset contains historical regimens (e.g., older sulfonylureas, troglitazone) that do not reflect contemporary 2026 pharmacotherapy.

---

### 2. Out-of-Network Readmission Censoring
- Readmission events are captured **only if the patient returned to one of the 130 participating Cerner-affiliated hospital facilities**.
- **Epidemiological Implication**: Patients who relocated, sought care at non-participating healthcare systems, or presented to un-affiliated hospital networks are recorded as `readmitted = 'NO'`. A `'NO'` label should not be interpreted as proof that no readmission occurred outside the participating hospital network.

---

### 3. Coarse Laboratory Granularity & Unmeasured Vitals
- Continuous numerical laboratory values are absent:
  - `A1Cresult` is categorized into only 4 discrete levels: `None`, `Norm`, `>7`, `>8`.
  - `max_glu_serum` is categorized into: `None`, `Norm`, `>200`, `>300`.
- Crucial physiological markers and vital signs are omitted from the database:
  - Continuous blood pressure (systolic/diastolic)
  - Body Mass Index (BMI) or height (since `weight` is 96.86% missing)
  - Renal panel values (estimated Glomerular Filtration Rate [eGFR], Serum Creatinine)
  - Lipid panels (LDL, HDL, Triglycerides)
  - Microvascular complication exams (urinary albumin-to-creatinine ratio)

---

### 4. Billing & Diagnostic Coding Artifacts (ICD-9-CM)
- Diagnoses are captured as ICD-9-CM billing codes rather than clinical narrative EHR notes or pathology reports.
- **Coding Behavior Bias**: Administrative coding is frequently optimized for hospital reimbursement (DRG grouping) rather than physiological precision. Secondary diagnoses beyond the top three (`diag_1`, `diag_2`, `diag_3`) are not individually itemized, though aggregate counts (`number_diagnoses`) are provided.

---

### 5. Absence of Outpatient Compliance & Social Determinants of Health (SDoH)
- The dataset does not track post-discharge medication adherence, dietary compliance, pharmacy refill rates, family caregiver support, or economic stability.
- While `payer_code` provides a coarse insurance proxy, true social determinants of health (housing stability, food security, transportation access) are unmeasured confounders.

---

### 6. Encounter-Level Clustering vs. Patient Independence
- As documented in `07_identifier_analysis.md`, 23.45% of patients contribute multiple encounters (up to 40).
- Modeling on un-grouped data violates the Independent and Identically Distributed (IID) assumption, requiring strict hierarchical or group-stratified experimental protocols.
