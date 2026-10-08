# Uncertainty × Missingness Interaction Analysis

## 1. Scientific Motivation

When a modality drops out, does the decision-level impact depend on whether the removed channel had **high certainty** (low $U_i$) versus **high uncertainty** (high $U_i$)?

To evaluate this interaction without parameter fitting, the frozen cohort ($N=500$) was stratified into low-uncertainty ($U_m \le Q_{25}$) and high-uncertainty ($U_m \ge Q_{75}$) strata for each removed modality.

---

## 2. Stratified Sensitivity Matrix

### A. Removing Retina (Evaluating `foot_clinical` vs Tri-Modal)
- Uncertainty Thresholds: $Q_{25} = 0.000305$, $Q_{75} = 0.001953$

| Stratum | Packets ($N$) | Mean $\Delta R$ | Mean $\Delta \text{DCRI}$ | Max $\Delta R$ |
| :--- | :---: | :---: | :---: | :---: |
| **Low Uncertainty Retina ($U_R \le Q_{25}$)** | 125 | **0.158412** | **0.158434** | 0.485120 |
| **High Uncertainty Retina ($U_R \ge Q_{75}$)** | 125 | **0.133205** | **0.133261** | 0.461102 |
| **Difference ($\Delta_{\text{Low}} - \Delta_{\text{High}}$)** | — | **$+0.025207$** | **$+0.025173$** | — |

### B. Removing Foot (Evaluating `retina_clinical` vs Tri-Modal)
- Uncertainty Thresholds: $Q_{25} = 0.488410$, $Q_{75} = 0.693147$

| Stratum | Packets ($N$) | Mean $\Delta R$ | Mean $\Delta \text{DCRI}$ | Max $\Delta R$ |
| :--- | :---: | :---: | :---: | :---: |
| **Low Uncertainty Foot ($U_F \le Q_{25}$)** | 125 | **0.118945** | **0.134120** | 0.514782 |
| **High Uncertainty Foot ($U_F \ge Q_{75}$)** | 125 | **0.089421** | **0.113204** | 0.442109 |
| **Difference ($\Delta_{\text{Low}} - \Delta_{\text{High}}$)** | — | **$+0.029524$** | **$+0.020916$** | — |

### C. Removing Clinical (Evaluating `retina_foot` vs Tri-Modal)
- Uncertainty Thresholds: $Q_{25} = 0.015625$, $Q_{75} = 0.054688$

| Stratum | Packets ($N$) | Mean $\Delta R$ | Mean $\Delta \text{DCRI}$ | Max $\Delta R$ |
| :--- | :---: | :---: | :---: | :---: |
| **Low Uncertainty Clinical ($U_C \le Q_{25}$)** | 125 | **0.068940** | **0.073210** | 0.289140 |
| **High Uncertainty Clinical ($U_C \ge Q_{75}$)** | 125 | **0.054320** | **0.060410** | 0.245100 |
| **Difference ($\Delta_{\text{Low}} - \Delta_{\text{High}}$)** | — | **$+0.014620$** | **$+0.012800$** | — |

---

## 3. Scientific Conclusions (H3 Empirically Supported Under Pre-Specified Stratification Analysis)

Across all three modality channels:
1. **Empirical Stratification Pattern**: Across the three modality-specific analyses, removal of low-uncertainty channels was associated with approximately **19–33% larger mean risk shifts** than removal of high-uncertainty channels:
   - Retina removal: $0.158412$ vs $0.133205$ ($+18.9\% \approx +19\%$)
   - Foot removal: $0.118945$ vs $0.089421$ ($+33.0\%$)
   - Clinical removal: $0.068940$ vs $0.054320$ ($+26.9\%$)
2. **Associational Mechanism**: Under ACARA-U routing ($z_i = \alpha C_i + \beta R_i - \gamma U_i + \eta Q_i$), lower uncertainty is associated with higher baseline authority ($w_i$). Consequently, removing a low-uncertainty modality removes a larger share of decision authority, resulting in a larger shift in the aggregated consensus. This is an observational stratified association; causal attribution is not established by this stratification alone.
