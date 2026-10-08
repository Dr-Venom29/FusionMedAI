# Experiment B: ACARA-U Routing Authority Response & Redistribution

## 1. Experimental Objective

Experiment B evaluates whether the frozen ACARA-U router ($z_i = \alpha C_i + \beta R_i - \gamma U_i + \eta Q_i$) dynamically attenuates decision authority ($w_i \downarrow$) when a modality's input signal degrades, and how the forfeited authority is redistributed across remaining channels.

Weight shift is defined as $\Delta w_i = w_i^{(d)} - \overline{w_i^{\text{clean}}}$, where the reference is the clean cohort mean authority prior to degradation.

---

## 2. Dynamic Routing Response Across Modalities

Across the frozen $N=500$ cohort ($\text{seed}=115$):

### 2.1 Retinal Fundus Routing Attenuation (D-R1 Gaussian Blur, $\overline{w_R^{\text{clean}}} = 0.5034$)
| Severity | Mean Weight ($\overline{w_R}$) | Weight Shift ($\Delta w_R$) | $95\%$ Paired Bootstrap CI | Relative Reduction ($\text{RAR}_R$) | Redistributed Authority |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **D0** | $0.5122$ | $+0.0089$ | $[+0.0088, +0.0089]$ | $-1.8\%$ | $-0.0089$ |
| **D1** | $0.3837$ | $-0.1196$ | $[-0.1203, -0.1189]$ | $24.3\%$ | $+0.1196$ |
| **D2** | $0.3581$ | $-0.1453$ | $[-0.1462, -0.1443]$ | $29.5\%$ | $+0.1453$ |
| **D3** | $0.3336$ | $\mathbf{-0.1697}$ | $[-0.1710, -0.1684]$ | $\mathbf{34.4\%}$ | $\mathbf{+0.1697}$ |

### 2.2 Diabetic Foot Ulcer Routing Attenuation (D-F1 Gaussian Blur, $\overline{w_F^{\text{clean}}} = 0.2568$)
| Severity | Mean Weight ($\overline{w_F}$) | Weight Shift ($\Delta w_F$) | $95\%$ Paired Bootstrap CI | Relative Reduction ($\text{RAR}_F$) | Redistributed Authority |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **D0** | $0.2393$ | $-0.0176$ | $[-0.0179, -0.0172]$ | $7.1\%$ | $+0.0176$ |
| **D1** | $0.1968$ | $-0.0601$ | $[-0.0613, -0.0588]$ | $24.1\%$ | $+0.0601$ |
| **D2** | $0.1433$ | $-0.1135$ | $[-0.1160, -0.1109]$ | $45.1\%$ | $+0.1135$ |
| **D3** | $0.1144$ | $\mathbf{-0.1425}$ | $[-0.1459, -0.1389]$ | $\mathbf{56.3\%}$ | $\mathbf{+0.1425}$ |

### 2.3 Structured Clinical EHR Routing Attenuation (D-C1 Random Masking, $\overline{w_C^{\text{clean}}} = 0.2398$)
| Severity | Mean Weight ($\overline{w_C}$) | Weight Shift ($\Delta w_C$) | $95\%$ Paired Bootstrap CI | Relative Reduction ($\text{RAR}_C$) | Redistributed Authority |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **D0** | $0.2677$ | $+0.0279$ | $[+0.0276, +0.0283]$ | $-11.9\%$ | $-0.0279$ |
| **D1** | $0.2358$ | $-0.0040$ | $[-0.0040, -0.0039]$ | $1.7\%$ | $+0.0040$ |
| **D2** | $0.1969$ | $-0.0428$ | $[-0.0434, -0.0423]$ | $18.1\%$ | $+0.0428$ |
| **D3** | $0.1483$ | $\mathbf{-0.0914}$ | $[-0.0928, -0.0901]$ | $\mathbf{38.5\%}$ | $\mathbf{+0.0914}$ |

---

## 3. Quality-Authority Response Slope ($S_{QW}$)

The empirical response slope $S_{QW} = \frac{\Delta w_i}{\Delta Q_i}$ measures the sensitivity of decision authority per unit quality degradation:

- **Retina (D-R1 Gaussian Blur)**: $S_{QW}^{\text{D3}} = \mathbf{0.2822}$ ($95\%$ CI: $[0.2799, 0.2842]$).
- **Foot (D-F1 Gaussian Blur)**: $S_{QW}^{\text{D3}} = \mathbf{0.1898}$ ($95\%$ CI: $[0.1850, 0.1943]$).
- **Clinical (D-C1 Random Mask)**: $S_{QW}^{\text{D3}} = \mathbf{0.1413}$ ($95\%$ CI: $[0.1393, 0.1434]$).

All slopes are strictly positive ($S_{QW} > 0$), confirming that authority loss scales directly with quality loss.

---

## 4. Authority Redistribution Conservation

Across all 1,500 modality-packet evaluations per single-modality severity condition (3 modalities $\times$ 500 packets):

$$
\sum_{j \ne i} \Delta w_j = -\Delta w_i \quad (\text{Exact within } 10^{-6})
$$

Simplex normalization $\sum_{i \in \mathcal{A}} w_i = 1.000000$ is strictly preserved under all degradation states.


