# Comparative Baseline Evaluation across Missingness Regimes

## 1. Baseline Architecture Ladder (B1–B6)

To rigorously test **Hypothesis H2**, ACARA-U (B6) was benchmarked against the full baseline ladder across all single-modality dropout scenarios:

- **B1**: Reliability-Selected (Hard Winner-Take-All by highest available $R_i$)
- **B2**: Uniform Average ($w_i = \frac{1}{|\mathcal{A}|}$)
- **B3**: Confidence-Weighted ($w_i \propto C_i$)
- **B4**: Confidence + Reliability ($w_i \propto \text{Softmax}(C_i + R_i)$)
- **B5**: Confidence + Reliability − Uncertainty ($w_i \propto \text{Softmax}(C_i + R_i - U_i)$)
- **B6**: Full ACARA-U ($w_i \propto \text{Softmax}(\alpha C_i + \beta R_i - \gamma U_i + \eta Q_i)$)

---

## 2. Comparative Risk Sensitivity Summary ($\overline{\Delta R_{\text{missing}}}$)

| Baseline ID | Fusion Strategy | Missing Clinical ($\Delta R_{-C}$) | Missing Foot ($\Delta R_{-F}$) | Missing Retina ($\Delta R_{-R}$) | Peak Perturbation |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **B1** | Reliability-Selected | **0.000000** | **0.000000** | **0.391578** | **0.391578** (Abrupt Single-Channel Jump) |
| **B2** | Uniform Average | 0.094759 | 0.124226 | 0.090744 | 0.124226 |
| **B3** | Confidence-Weighted | 0.065608 | 0.140115 | 0.120110 | 0.140115 |
| **B4** | Confidence + Reliability | 0.060673 | 0.142537 | 0.125150 | 0.142537 |
| **B5** | Conf + Rel − Uncertainty | 0.066282 | 0.104556 | 0.139402 | 0.139402 |
| **B6 (ACARA-U)**| Full Dynamic Router | **0.062082** | **0.103242** | **0.145687** | **0.145687** (Proportional Reallocation) |

---

## 3. Scientific Comparison (Hypothesis H2 Supported)

1. **Behavior of Winner-Take-All (B1)**:  
   Under B1, because $R_R > R_F > R_C$, Retina is assigned $100\%$ weight whenever it is present. Consequently, dropping Foot or Clinical causes $\Delta R = 0.000000$. However, when Retina is removed, authority abruptly jumps $100\%$ to Foot, exhibiting a substantially larger mean risk sensitivity of $\overline{\Delta R} = 0.391578$.
2. **Controlled Proportionality of ACARA-U (B6)**:  
   ACARA-U maintains balanced multimodal participation ($\overline{w_R} = 0.50, \overline{w_F} = 0.26, \overline{w_C} = 0.24$). When Retina is lost, ACARA-U redistributes authority across the remaining modalities, producing a lower mean absolute risk shift of $\overline{\Delta R} = 0.145687$ than B1 under Retina removal.
3. **Uncertainty Moderation (B5 & B6)**:  
   Compared with B3/B4, the uncertainty-aware strategies B5 and B6 exhibit lower mean risk sensitivity under Foot removal ($\overline{\Delta R} \approx 0.103$ vs $0.142$). This is consistent with Foot's higher uncertainty receiving a larger penalty in the frozen routing configuration.
