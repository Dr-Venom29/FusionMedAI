# Availability Regimes Performance Matrix

## 1. Availability Configuration Taxonomy

The multimodal architecture evaluates decision packets across $2^3 = 8$ availability states:

| Regime Key | Active Modalities ($M$) | Description | Status |
| :--- | :---: | :--- | :--- |
| `tri_modal` | 3 | Full Tri-Modal Availability ($\text{Retina} + \text{Foot} + \text{Clinical}$) | `SUCCESS` |
| `retina_foot` | 2 | Missing Clinical Channel ($-C$) | `SUCCESS` |
| `retina_clinical` | 2 | Missing Foot Channel ($-F$) | `SUCCESS` |
| `foot_clinical` | 2 | Missing Retina Channel ($-R$) | `SUCCESS` |
| `retina_only` | 1 | Unimodal Retinal Image Only | `SUCCESS` |
| `foot_only` | 1 | Unimodal Diabetic Foot Image Only | `SUCCESS` |
| `clinical_only` | 1 | Unimodal Tabular Clinical Features Only | `SUCCESS` |
| `zero_modality` | 0 | All Modalities Unavailable ($\emptyset$) | `NO_MODALITY_AVAILABLE` |

---

## 2. Empirical Performance Summary ($N=500, \text{seed}=115$)

| Regime | Active ($M$) | Mean $R_{\text{fusion}}$ | Mean $\text{DCRI}_{0.20}$ | Mean $U_{\text{sum}}$ | Mean Entropy $H(w)$ | Mean $w_R$ | Mean $w_F$ | Mean $w_C$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`tri_modal`** | 3 | 0.288499 | 0.161712 | 0.633936 | 1.009745 | 0.503350 | 0.256884 | 0.239766 |
| **`retina_foot`** | 2 | 0.345801 | 0.226742 | 0.595293 | 0.632924 | 0.662333 | 0.337667 | 0.000000 |
| **`retina_clinical`** | 2 | 0.199731 | 0.191728 | 0.040018 | 0.627725 | 0.674925 | 0.000000 | 0.325075 |
| **`foot_clinical`** | 2 | 0.332652 | 0.206140 | 0.632561 | 0.690858 | 0.000000 | 0.508484 | 0.491516 |
| **`retina_only`** | 1 | 0.255037 | 0.254762 | 0.001375 | 0.000000 | 1.000000 | 0.000000 | 0.000000 |
| **`foot_only`** | 1 | 0.521907 | 0.403124 | 0.593918 | 0.000000 | 0.000000 | 1.000000 | 0.000000 |
| **`clinical_only`** | 1 | 0.113468 | 0.105739 | 0.038643 | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
| **`zero_modality`** | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

---

## 3. Key Observations

1. **Unimodal Simplex Collapse**: In unimodal regimes (`retina_only`, `foot_only`, `clinical_only`), the available channel automatically receives $w = 1.000000$ and routing entropy collapses to $H = 0.000000$.
2. **Observed Entropy Ordering**: In the frozen C11.8 cohort, mean routing entropy increased with the number of available modalities:

$$H(M=1) = 0.000000 < H(M=2) \approx 0.65 < H(M=3) \approx 1.0097$$

This is an empirical property of the evaluated cohort, not a general mathematical guarantee of the routing mechanism.
3. **Fail-Closed Protection**: In `zero_modality`, all weights and risk outputs evaluate strictly to $0.000000$ with status `NO_MODALITY_AVAILABLE`.
