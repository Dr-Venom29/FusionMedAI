# Comparative Baseline Benchmarking & Quality Isolation

## 1. Experimental Objective

To isolate the scientific role of the explicit quality input $Q_i$ in dynamic routing, ACARA-U (B6) is benchmarked against the standard experimental ladder (B1–B5) under identical degradation conditions:

$$
\begin{aligned}
\text{B1 (Reliability-Selected)}&: z_i = R_i \implies \text{Argmax}(R) \\
\text{B2 (Uniform Average)}&: z_i = 0 \implies w_i = 1 / |\mathcal{A}| \\
\text{B3 (Confidence-Weighted)}&: z_i = 1.0 \cdot C_i \\
\text{B4 (Confidence + Reliability)}&: z_i = 1.0 \cdot C_i + 1.5 \cdot R_i \\
\text{B5 (Conf + Rel } - \text{ Uncertainty)}&: z_i = 1.0 \cdot C_i + 1.5 \cdot R_i - 1.0 \cdot U_i \\
\text{B6 (Full ACARA-U)}&: z_i = 1.0 \cdot C_i + 1.5 \cdot R_i - 1.0 \cdot U_i + 0.5 \cdot Q_i
\end{aligned}
$$

The primary comparative test is **B6 vs B5**, as the only architectural distinction between them is the presence of signal quality $Q_i$.

---

## 2. Comparative Authority Attenuation Under Severe Degradation (D3)

![Figure 10.4: Isolating Quality Contribution: B6 vs B5 Under Retinal Degradation](figures/fig10_4_b5_vs_b6_quality_isolation.png)

Evaluation on Retinal Fundus Blur (D-R1) across the frozen $N=500$ cohort ($\text{seed}=115$):

| Baseline | Strategy Formulation | D0 Clean $w_R$ | D3 Severe $w_R$ | Authority Shift ($\Delta w_R$) | Quality Awareness Mechanism |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **B1** | Reliability-Selected | $1.0000$ | $1.0000$ | $\mathbf{0.0000}$ | Completely insensitive to signal quality |
| **B2** | Uniform Average | $0.3333$ | $0.3333$ | $\mathbf{0.0000}$ | Fixed static allocation |
| **B3** | Confidence-Weighted | $0.3944$ | $0.3895$ | $\mathbf{-0.0049}$ | Weak indirect attenuation via confidence |
| **B4** | Confidence + Reliability | $0.4049$ | $0.3999$ | $\mathbf{-0.0050}$ | Weak indirect attenuation |
| **B5** | Conf + Rel $-$ Uncertainty | $0.4852$ | $0.4464$ | $\mathbf{-0.0388}$ | Moderate attenuation via uncertainty boost |
| **B6** | **Full ACARA-U** | $0.5122$ | $0.3336$ | $\mathbf{-0.1697}$ | **Direct quality attenuation ($Q_R \downarrow \implies w_R \downarrow$)** |

---

## 3. Isolating the Contribution of $Q_i$ (B6 vs B5 Paired Analysis)

The paired packet-level difference in authority reduction is defined as:

$$
D^{(\text{B6}-\text{B5})} = \Delta w_i^{(\text{B6})} - \Delta w_i^{(\text{B5})}
$$

- **D1 (Mild Blur)**: $D^{(\text{B6}-\text{B5})} = \mathbf{-0.1113}$ ($95\%$ CI: $[-0.1119, -0.1106]$).
- **D2 (Moderate Blur)**: $D^{(\text{B6}-\text{B5})} = \mathbf{-0.1258}$ ($95\%$ CI: $[-0.1266, -0.1250]$).
- **D3 (Severe Blur)**: $D^{(\text{B6}-\text{B5})} = \mathbf{-0.1309}$ ($95\%$ CI: $[-0.1319, -0.1300]$).

### Scientific Conclusion
Because the $95\%$ paired bootstrap confidence interval strictly excludes zero at all progressive degradation levels ($[-0.1319, -0.1300]$ at D3), the paired comparison showed significantly greater authority attenuation under B6 than B5 ($4.3\times$ greater attenuation, paired difference $D = -0.1309$, $95\%$ bootstrap CI: $[-0.1319, -0.1300]$), supporting the incremental contribution of the quality term within the tested controlled degradation benchmark.



