# Comprehensive Results Summary & Hypothesis Outcomes

## 1. Primary Empirical Findings ($N=500, \text{seed}=115$)

### A. Availability Regimes Performance Matrix

| Regime | Active Count ($M$) | Mean $R_{\text{fusion}}$ | Mean $\text{DCRI}_{0.20}$ | Mean $U_{\text{sum}}$ | Mean Entropy $H(w)$ | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`tri_modal`** | 3 | 0.288499 | 0.161712 | 0.633936 | 1.009745 | `SUCCESS` |
| **`retina_foot`** | 2 | 0.345801 | 0.226742 | 0.595293 | 0.632924 | `SUCCESS` |
| **`retina_clinical`** | 2 | 0.199731 | 0.191728 | 0.040018 | 0.627725 | `SUCCESS` |
| **`foot_clinical`** | 2 | 0.332652 | 0.206140 | 0.632561 | 0.690858 | `SUCCESS` |
| **`retina_only`** | 1 | 0.255037 | 0.254762 | 0.001375 | 0.000000 | `SUCCESS` |
| **`foot_only`** | 1 | 0.521907 | 0.403124 | 0.593918 | 0.000000 | `SUCCESS` |
| **`clinical_only`** | 1 | 0.113468 | 0.105739 | 0.038643 | 0.000000 | `SUCCESS` |
| **`zero_modality`** | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | `NO_MODALITY_AVAILABLE` |

---

### B. Single-Modality Dropout Sensitivity ($\Delta R$)

$$\Delta R_{\text{missing}} = |R_{\text{fusion}}^{\text{full}} - R_{\text{fusion}}^{\text{subset}}|$$

- **Missing Clinical (RF)**: Mean $\Delta R = 0.062082 \pm 0.052183$ (Median: $0.046397$, Range: $[0.000102, 0.312953]$, $95\%\text{ CI: } [0.057628, 0.066712]$)
- **Missing Foot (RC)**: Mean $\Delta R = 0.103242 \pm 0.094595$ (Median: $0.076063$, Range: $[0.000052, 0.514782]$, $95\%\text{ CI: } [0.095084, 0.111734]$)
- **Missing Retina (FC)**: Mean $\Delta R = 0.145687 \pm 0.097022$ (Median: $0.125667$, Range: $[0.000098, 0.505470]$, $95\%\text{ CI: } [0.137358, 0.154205]$)

---

## 2. Hypothesis Testing Outcomes

| Hypothesis | Registered Statement | Verification Outcome | Empirical Evidence |
| :--- | :--- | :---: | :--- |
| **H1 (Availability Safety & Simplex Conservation)** | Router enforces $A_i=0 \implies w_i=0.0$ and $\sum w_i=1.0$ across all non-empty subsets. | **CONFIRMED / VERIFIED** | 0 availability violations across 4,000 evaluations; 7,500/7,500 masked-invariance checks passed. |
| **H2 (Comparative Missing-Modality Sensitivity)** | ACARA-U exhibits distinct and empirically bounded decision-level risk sensitivity under controlled modality loss compared with the evaluated baseline strategies. | **SUPPORTED** | B1 exhibits substantially larger mean risk sensitivity ($\Delta R = 0.3916$ on Retina loss), while ACARA-U redistributes authority with lower absolute shift ($\Delta R = 0.1457$). |
| **H3 (Uncertainty-Stratified Missingness Sensitivity)** | The decision-level impact of modality removal differs between low- and high-uncertainty strata, indicating an association between instance-level predictive uncertainty and missing-modality sensitivity. | **SUPPORTED** | Across modality-specific analyses, low-uncertainty channel removal produced larger mean risk shifts than high-uncertainty channel removal (Retina: $0.1584$ vs $0.1332$, Foot: $0.1189$ vs $0.0894$, Clinical: $0.0689$ vs $0.0543$; relative differences $\approx 19–33\%$; observational association). |
