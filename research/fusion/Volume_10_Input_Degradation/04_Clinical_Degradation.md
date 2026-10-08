# Structured Clinical EHR Input Degradation Operators

## 1. Modality Degradation Taxonomy

The Structured Clinical EHR channel ($\mathcal{M}_{\text{Clinical}}$) is subjected to tabular feature corruptions reflecting missing documentation, unperformed lab panels, and EHR data entry errors across the canonical 119-D feature contract:

1. **D-C1: Random Feature Masking (`OP_CLINICAL_RANDOM_MASK`)**  
   Simulates unrecorded data fields by randomly setting a specified fraction of valid features to missing (`NaN`).
   
2. **D-C2: Structured Feature Masking (`OP_CLINICAL_STRUCTURED_MASK`)**  
   Simulates missing clinical domains (e.g., missing admission context, omitted laboratory panel, or unrecorded inpatient medications).
   
3. **D-C3: Controlled Value Perturbation (`OP_CLINICAL_PERTURBATION`)**  
   Simulates continuous measurement noise and physiological telemetry drift by injecting additive Gaussian noise $\mathcal{N}(0, \sigma^2)$ into continuous features; values exceeding normalized validity limits ($|z| > 3.5$) are invalidated.
   
4. **D-C4: Diagnostic Domain Omission (`OP_CLINICAL_DOMAIN_OMISSION`)**  
   Simulates complete absence of entire hospital sub-specialty evaluation blocks.

---

## 2. Canonical Clinical Feature Groups (119-D Representation)

| Feature Group | Dimension Range | Clinical Content Description |
| :--- | :---: | :--- |
| **Demographics** | $[0, 15)$ | Age bins, gender, race, admission type |
| **Admission Context** | $[15, 35)$ | Time in hospital, discharge disposition, admission source |
| **Laboratory Panels** | $[35, 65)$ | Lab procedures count, HbA1c status, blood glucose categories |
| **Medications** | $[65, 95)$ | Insulin, metformin, sulfonylureas, DPP-4 inhibitors |
| **Diagnoses & Utilization** | $[95, 119)$ | Primary/secondary ICD-9 categories, emergency visits, prior admissions |

---

## 3. Parameterization Across the 4-Level Severity Grid

| Severity Level | State Description | D-C1 Mask Fraction | D-C2 Omitted Groups | D-C3 Noise $\sigma$ (Fraction) | D-C4 Omitted Domains |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **D0** | Clean Baseline | $0.00$ (0 / 119) | None | $\sigma=0.00$ ($0\%$) | 0 Domains |
| **D1** | Mild Degradation | $0.15$ (18 / 119) | 1 Group | $\sigma=0.50$ ($15\%$) | 1 Domain |
| **D2** | Moderate Degradation | $0.35$ (42 / 119) | 2 Groups | $\sigma=1.50$ ($35\%$) | 2 Domains |
| **D3** | Severe Degradation | $0.65$ (77 / 119) | 3 Groups | $\sigma=3.00$ ($65\%$) | 3 Domains |


---

## 4. Signal Quality Layer Coupling

Degraded feature vectors $x_C^{(d)}$ are evaluated directly by the frozen clinical quality engine:

$$
Q_C(x_C^{(d)}) = \frac{N_{\text{valid}}(x_C^{(d)})}{119}
$$

where $N_{\text{valid}}$ counts features strictly meeting valid bounds without missingness or NaN values.
