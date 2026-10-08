# Formal Clinical & Methodological Boundary Declaration

## 1. Unpaired Multimodal Data Architecture

The FusionMedAI research pipeline operates across three independently curated clinical validation cohorts:
1. **Retinal Fundus Image Cohort** (5-Class Diabetic Retinopathy)
2. **Diabetic Foot Photograph Cohort** (4-Class Wagner Ulcer Grading)
3. **Structured Tabular EHR Cohort** (119-D Binary 30-Day Hospital Readmission Risk)

These datasets are **unpaired**: no single human patient has simultaneous ground-truth records across all three channels.

---

## 2. Explicit Scientific Boundaries

To preserve strict scientific integrity, the following negative constraints are permanently enforced:

1. **No Fabricated Multimodal Patient Ground Truth**:
   We explicitly refuse to fabricate synthetic patient labels ($Y_{\text{fake}} \in \{0, 1\}$) to construct joint ROC curves, Precision-Recall curves, or calibration curves for multimodal fusion.
2. **No Fusion-Level Calibration Claims**:
   We do not claim that $R_{\text{fusion}}$ represents a "clinically calibrated probability." Reporting metrics such as "Fusion ECE = X" or "Fusion Brier = X" without a true paired multimodal test set constitutes invalid scientific methodology.
3. **Derived Decision Index Semantics**:
   $R_{\text{fusion}}$ is mathematically defined as a **decision-level composite risk index**:

   $$
   R_{\text{fusion}} = \sum_{i \in \mathcal{A}} w_i \cdot r_i
   $$

   It measures the consensus severity projection weighted by model reliability, confidence, uncertainty, and quality.

---

## 3. Legitimate Scope of Phase C11.11

Within these boundaries, Phase C11.11 legitimately establishes:
- Modality-level probability quality improvements under validation ground truth (ECE, Brier, NLL reductions).
- Impact of probability calibration on scalar risk projections ($r_i$) and confidence scores ($C_i$).
- Sensitivity of ACARA-U dynamic routing authority to calibrated versus uncalibrated inputs.
- Conservation of the routing simplex ($\sum w_i = 1.0$) and stability of routing entropy under calibrated representations.
- Persistence of calibration adjustments across progressive input degradation ladders ($D0 \to D3$).
