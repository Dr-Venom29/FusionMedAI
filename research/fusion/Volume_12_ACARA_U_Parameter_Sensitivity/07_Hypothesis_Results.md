# Hypothesis Evaluation & Scientific Verdicts

## 1. Summary of Scientific Outcomes

Phase C11.12 pre-specified eight hypotheses to rigorously evaluate parameter sensitivity, invariant conservation, and local neighborhood stability.

### Table 12.5: Hypothesis Evaluation Scoreboard

| Hypothesis | Description | Target Term | Status | Empirical Evidence |
| :--- | :--- | :---: | :---: | :--- |
| **H1** | Confidence Sensitivity | $\alpha$ | **Supported** | $S_\alpha(w_R) = +0.062972$, $S_\alpha(w_C) = -0.086642$, logit derivative max error $= 8.9\times 10^{-16}$ |
| **H2** | Reliability Sensitivity | $\beta$ | **Supported** | $S_\beta(w_R) = +0.013270$, $S_\beta(w_C) = -0.018214$, higher validation reliability favored, logit error $= 1.2\times 10^{-15}$ |
| **H3** | Uncertainty Sensitivity | $\gamma$ | **Supported** | Foot uncertainty attenuation slope $= -0.099411$, logit derivative max error $= 7.2\times 10^{-16}$, zero collapse |
| **H4** | Quality Sensitivity | $\eta$ | **Supported** | Retina quality bonus slope $= +0.017998$, logit derivative max error $= 5.6\times 10^{-16}$ |
| **H5** | Invariant Preservation | All | **Confirmed** | $100.0\%$ of packets satisfy $\sum_{i \in \mathcal{A}} w_i = 1.000000$ and $w_i \ge 0.0$ across all 23 evaluations |
| **H6** | Missing-Modality Invariance | All | **Confirmed** | Inactive channel perturbations yield exactly $0.000000$ cross-talk across all 3 channels ($A_R, A_F, A_C = 0$) |
| **H7** | Reference Reproducibility | $\Theta_0$ | **Confirmed** | $\Theta_0$ reproduces sealed baseline values ($\bar{w}_R=0.501002, \bar{w}_F=0.260925, \bar{w}_C=0.238073$) |
| **H8** | Local Behavioral Stability | All | **Supported** | No routing collapse observed; entropy maintained in $[0.9633, 1.0460]\text{ nats}$ ($> 0.85$ threshold) across all 23 evaluations |

---

## 2. Detailed Hypothesis Analysis

### H1: Confidence Sensitivity ($\alpha$) — Supported
- **Mechanism**: Higher $\alpha$ amplifies the influence of per-case model confidence $C_i$.
- **Evidence**: Retinal authority $\bar{w}_R$ rises from $0.4673$ ($\alpha=0.50$) to $0.5303$ ($\alpha=1.50$), while Clinical authority $\bar{w}_C$ attenuates from $0.2846$ to $0.1980$, aligning with the higher average confidence of the retinal image backbone. Analytical derivative check passed with max error $8.9\times 10^{-16}$.

### H2: Reliability Sensitivity ($\beta$) — Supported
- **Mechanism**: Higher $\beta$ weights the static validation reliability prior ($R_R=0.929956 > R_F=0.922266 > R_C=0.825382$).
- **Evidence**: Scaling $\beta$ from $1.00$ to $2.00$ monotonically transfers authority from Clinical ($\Delta w_C = -0.0182$) to Retina ($\Delta w_R = +0.0133$), consistent with the ordering of the frozen reliability priors. Analytical derivative check passed with max error $1.2\times 10^{-15}$.

### H3: Uncertainty Sensitivity ($\gamma$) — Supported
- **Mechanism**: Higher $\gamma$ penalizes channels with high predictive uncertainty $U_i$.
- **Evidence**: Scaling $\gamma$ from $0.50$ to $1.50$ produces a substantial $-0.0994$ authority reduction on the Foot channel (highest average uncertainty), conservatively absorbed by Retina ($+0.0723$) and Clinical ($+0.0271$). Analytical derivative check passed with max error $7.2\times 10^{-16}$.

### H4: Quality Sensitivity ($\eta$) — Supported
- **Mechanism**: Scales input signal quality bonus $Q_i$.
- **Evidence**: Scaling $\eta$ awards a positive authority slope to clean retinal images ($S_\eta(w_R) = +0.0180$). Across all $N=500$ packets, empirical logit differences match the theoretical prediction $\Delta(z_i - z_j) = \Delta \eta (Q_i - Q_j)$ with machine precision ($5.6\times 10^{-16}$).

### H5 & H6: Structural Invariants — Confirmed
- Zero violations of the probability simplex or non-negativity were detected across $23 \times 500 = 11,500$ packet evaluations.
- Missing-modality hard-masking was exhaustively verified across all three individual channel dropout conditions ($A_R=0, A_F=0, A_C=0$), completely isolating active routing from inactive channel corruptions (max cross-talk $= 0.0\times 10^0$).

### H7 & H8: Reference Stability & Absence of Collapse — Supported
- The reference configuration $\Theta_0$ reproduces sealed baseline values within numerical tolerances.
- Across the entire tested perturbation domain, the ACARA-U router maintains high entropy ($\bar{H} \ge 0.9633\text{ nats}$), supporting the absence of routing-collapse or single-channel saturation behavior within the tested perturbation domain.
