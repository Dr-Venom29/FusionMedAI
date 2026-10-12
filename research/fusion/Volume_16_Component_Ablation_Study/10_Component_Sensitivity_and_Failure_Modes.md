# Chapter 10: Component Sensitivity and Failure Modes

## 1. Analysis of Systemic Sensitivities

Component ablation reveals specific vulnerabilities and operational failure modes that arise when individual components of the ACARA-U architecture are removed or perturbed.

---

## 2. Failure Mode Catalog

### 2.1 Failure Mode 1: Over-Reliance on High-Uncertainty Channels (Ablating $\gamma$)
- **Trigger:** Setting $\gamma = 0$ (Variant $A3$).
- **Mechanism:** In the absence of the uncertainty penalty $-\gamma U_i$, the router treats model outputs as equally reliable regardless of epistemic confidence.
- **Manifestation:** In the benchmark cohort, foot ulcer model outputs carry high epistemic variance ($\overline{U_{\text{foot}}} \approx 0.42$). Without $\gamma$, foot ulcer weight surges by $+0.116634$, inflating fused risk by $+0.030842$ and causing $15$ additional Tier 2 review escalations.
- **Mitigation:** Retaining $\gamma = 1.0$ is essential to safeguard against uncalibrated out-of-distribution vision model predictions.

### 2.2 Failure Mode 2: Loss of Modality Specificity (Ablating $\alpha$)
- **Trigger:** Setting $\alpha = 0$ (Variant $A1$).
- **Mechanism:** Disabling dynamic confidence forces the router to rely solely on static historical reliability priors ($R_i$) and quality scores ($Q_i$).
- **Manifestation:** The router cannot reward clear, unambiguous diagnostic signals (e.g., severe proliferative diabetic retinopathy with $C_{\text{retina}} \to 1.0$). Consequently, $24$ encounters previously categorized in Tier 2 (Escalation) are downgraded to Tier 1 (Additional Assessment).
- **Mitigation:** Dynamic confidence weighting ($\alpha = 1.0$) is required to dynamically respond to acute findings in individual channels.

### 2.3 Failure Mode 3: Negative DCRI Anomaly (Retaining $\delta$ without Floor)
- **Trigger:** Subtraction of linear uncertainty penalty $R_{\text{fusion}} - \delta^* \sum_{i \in \mathcal{A}} U_i$ when $R_{\text{fusion}}$ is near zero.
- **Mechanism:** For encounters with low baseline risk ($R_{\text{fusion}} < 0.05$) and moderate model uncertainty ($U_i > 0.50$), subtracting $0.10 \times 0.50 = 0.05$ produces negative composite scores.
- **Manifestation:** Exactly $37/500$ packets ($7.4\%$) in reference architecture $A0$ produce negative DCRI values (minimum observed: $-0.027143$; mean of negative values: $-0.010750$).
- **Mitigation & Future Recommendation:** In Phase C11.17, evaluate implementing a rectified zero-floor:

$$
\mathrm{DCRI}_{\mathrm{rectified}} = \max\left(0.0, R_{\mathrm{fusion}} - \delta^* \sum_{i \in \mathcal{A}} U_i\right)
$$

---

## 3. Boundary & Decoupling Invariants

### 3.1 Weight-Risk Decoupling Theorem
As demonstrated in Volume 15 and confirmed in Chapter 05, increasing the routing weight $w_i$ of a channel does not guarantee an increase in fused risk $R_{\mathrm{fusion}}$:

$$
\frac{\partial R_{\mathrm{fusion}}}{\partial z_i} = w_i \left( r_i - \sum_{j \in \mathcal{A}} w_j r_j \right)
$$

The directional response is strictly governed by whether modality risk $r_i$ exceeds the weighted average of other channels.

### 3.2 Error and Corruption Resilience
Across all 7 accepted variants ($A0$–$A5$, $A_{\text{UNIFORM}}$) and all 4,000 evaluations per variant ($28,000$ total evaluations):
- **Packet Exceptions:** $0 / 28,000$ ($0.0\%$)
- **Non-Finite (NaN / Inf) Outputs:** $0 / 28,000$ ($0.0\%$)
- **Simplex Deviations ($|\sum w_i - 1| > 10^{-6}$):** $0 / 28,000$ ($0.0\%$)
- **Masking Violations ($w_{\text{inactive}} > 10^{-6}$):** $0 / 28,000$ ($0.0\%$)

All accepted variants execute within safe bounds.
