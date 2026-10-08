# Cross-Modality Conflict & Fused Risk Distribution

## 1. Cross-Modality Conflict Metrics by Combination

Cross-modality conflict is evaluated using two primary metrics:
1. **Maximum Pairwise Risk Disagreement**:
   
$$
\Delta_{\max} = \max_{i, j \in \mathcal{A}} |r_i - r_j| \quad (\text{for } |\mathcal{A}| \ge 2, \text{ else } 0.0)
$$

2. **Weighted Probability Dispersion**:
   
$$
\sigma_w = \sqrt{\sum_{i \in \mathcal{A}} w_i (r_i - R_{\text{fusion}})^2}
$$

### Empirical Conflict Profile by Combination (D2 Cohort)

| Combination | Modality Set | Mean $\Delta_{\max}$ | Std $\Delta_{\max}$ | Mean $\sigma_w$ | Conflict Rate ($\Delta_{\max} > 0.30$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RFC** | Tri-modal | **0.5404** | $0.2115$ | **0.2512** | **84.0%** |
| **RF** | Bimodal | **0.3857** | $0.2458$ | **0.1915** | **59.2%** |
| **RC** | Bimodal | **0.2915** | $0.2284$ | **0.1412** | **42.7%** |
| **FC** | Bimodal | **0.4285** | $0.2391$ | **0.2108** | **68.0%** |
| **R** | Unimodal | **0.0000** | $0.0000$ | **0.0000** | **0.0%** |
| **F** | Unimodal | **0.0000** | $0.0000$ | **0.0000** | **0.0%** |
| **C** | Unimodal | **0.0000** | $0.0000$ | **0.0000** | **0.0%** |

---

## 2. Risk Shift Under Modality Reduction

For every packet $k$, the risk shift relative to its reference tri-modal fusion is defined as:

$$
\Delta R = |R_{\text{fusion}}(k) - R_{\text{fusion}}^{\text{RFC}}(k)|
$$

### Risk Shift Progression (D2 Cohort)
- **RFC**: $\overline{\Delta R} = 0.0000 \pm 0.0000$ ($95\%$ CI: $[0.0, 0.0]$) — Identity reference.
- **RF (Missing Clinical)**: $\overline{\Delta R} = 0.0625 \pm 0.0520$ ($95\%$ CI: $[0.0533, 0.0715]$) — Smallest bimodal perturbation.
- **RC (Missing Foot)**: $\overline{\Delta R} = 0.1084 \pm 0.0982$ ($95\%$ CI: $[0.0855, 0.1314]$) — Intermediate bimodal perturbation.
- **FC (Missing Retina)**: $\overline{\Delta R} = 0.1132 \pm 0.0725$ ($95\%$ CI: $[0.0940, 0.1333]$) — Largest bimodal perturbation.
- **R (Unimodal Retina)**: $\overline{\Delta R} = 0.1618 \pm 0.1172$ ($95\%$ CI: $[0.1197, 0.2039]$).
- **F (Unimodal Foot)**: $\overline{\Delta R} = 0.2109 \pm 0.1584$ ($95\%$ CI: $[0.1463, 0.2754]$).
- **C (Unimodal Clinical)**: $\overline{\Delta R} = 0.2022 \pm 0.1398$ ($95\%$ CI: $[0.1408, 0.2637]$).

### Findings
1. Risk shift increased consistently from tri-modal to bimodal to unimodal configurations in the observed D2 cohort.
2. The observed sensitivity pattern is consistent with the frozen reliability ranking: removing Retina produced the largest observed bimodal perturbation ($\overline{\Delta R} = 0.1132$ in FC vs $0.0625$ in RF and $0.1084$ in RC), while unimodal Foot ($0.2109$) and Clinical ($0.2022$) produced larger shifts than unimodal Retina ($0.1618$).
