# Reference Configuration Characterization ($\Theta_0$)

## 1. Reference Architecture & Coefficient Specification

The frozen reference router configuration $\Theta_0$ is defined as:

$$\Theta_0 = (\alpha=1.00, \beta=1.50, \gamma=1.00, \eta=0.50)$$

coupled to the provisional decision-level uncertainty penalty parameter $\delta = 0.20$.

The coefficient weighting follows the frozen design hierarchy established in the preceding ACARA-U development phases:
1. **Primary Reliability Anchor ($\beta = 1.50$)**: Reflects the long-term empirical validation performance of the constituent backbones ($R_R=0.929956, R_F=0.922266, R_C=0.825382$).
2. **Dynamic Confidence Modulation ($\alpha = 1.00$)**: Adapts per-case authority to model certainty ($C_i$).
3. **Uncertainty Risk Penalty ($\gamma = 1.00$)**: Attenuates channels with elevated predictive entropy ($U_i$).
4. **Input Quality Bonus ($\eta = 0.50$)**: Provides adaptive authority modulation responding to raw signal degradation ($Q_i$).

---

## 2. Empirical Performance of $\Theta_0$ on the Frozen Cohort ($N=500$)

When evaluated on the frozen benchmark cohort ($N=500, \text{seed}=115$) under full tri-modal availability (RFC), $\Theta_0$ yields the following baseline metrics:

### Modality Authority Allocation
- **Retinal Authority ($\bar{w}_R$)**: $0.501002 \pm 0.088257$ (Range: $[0.232546, 0.658518]$)
- **Foot Authority ($\bar{w}_F$)**: $0.260925 \pm 0.086361$ (Range: $[0.118652, 0.536269]$)
- **Clinical Authority ($\bar{w}_C$)**: $0.238073 \pm 0.052254$ (Range: $[0.145064, 0.409576]$)

### Information Diversity & Dominance
- **Mean Routing Entropy ($\bar{H}$)**: $1.011041 \pm 0.051914\text{ nats}$ (Max Simplex Entropy $\ln(3) \approx 1.098612$)
- **Dominant Modality Rate**:
  - Retina Dominant ($\text{argmax } w_i = \text{Retina}$): $91.2\%$ ($456/500$)
  - Foot Dominant: $7.4\%$ ($37/500$)
  - Clinical Dominant: $1.4\%$ ($7/500$)

### Aggregated Decision Indices & Conflict
- **Mean Fused Risk ($\bar{R}_{\text{fusion}}$)**: $0.289900 \pm 0.163471$ (Range: $[0.011748, 0.792502]$)
- **Mean Provisional DCRI ($\overline{\text{DCRI}}$)**: $0.163113 \pm 0.184598$ (Range: $[-0.124183, 0.727629]$)
- **Mean Pairwise Conflict ($\bar{\Delta}_{\max}$)**: $0.520331 \pm 0.245812$
- **Weighted Dispersion ($\bar{\sigma}_w$)**: $0.215476 \pm 0.103982$

---

## 3. Stability Rationale

The reference point $\Theta_0$ provides high-entropy, non-degenerate routing ($\bar{H} \approx 1.011\text{ nats}$, approximately $92.0\%$ of the theoretical maximum), while retaining non-zero authority across all available modalities and preserving responsiveness to channel degradation.
