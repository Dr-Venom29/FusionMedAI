# Chapter 05: Fused Risk Propagation

## 1. Mathematical Mechanics of Risk Propagation

Fused risk is computed as the inner product of routing weights and modality risk predictions:

$$
R_{\mathrm{fusion}} = \sum_{i \in \mathcal{A}} w_i r_i
$$

When an ablation alters channel weights by $\Delta w_i = w_i^{(Av)} - w_i^{(A0)}$, the resulting shift in fused risk is:

$$
\Delta R_{\mathrm{fusion}} = R_{\mathrm{fusion}}^{(Av)} - R_{\mathrm{fusion}}^{(A0)} = \sum_{i \in \mathcal{A}} \Delta w_i r_i
$$

Because $\sum_{i \in \mathcal{A}} \Delta w_i = 0$, the change in fused risk can be expressed as:

$$
\Delta R_{\mathrm{fusion}} = \sum_{i \in \mathcal{A}} \Delta w_i (r_i - \bar{r}_{\mathcal{A}}), \quad \text{where} \quad \bar{r}_{\mathcal{A}} = \frac{1}{\lvert \mathcal{A} \rvert} \sum_{j \in \mathcal{A}} r_j
$$

is the unweighted mean risk across active modalities $\mathcal{A}$.

Therefore, whether an ablation increases or decreases fused risk depends critically on the correlation between weight changes $\Delta w_i$ and channel risk levels $r_i$:
- Reallocating weight toward a higher-risk modality increases fused risk ($\Delta R > 0$).
- Reallocating weight toward a lower-risk modality decreases fused risk ($\Delta R < 0$).

---

## 2. Paired Fused Risk Shifts by Variant

The table below details the paired fused risk shifts across the $N=500$ cohort:

| Variant | Target Component | Reference Mean $\overline{R}$ | Ablated Mean $\overline{R}$ | Paired Difference $\overline{\Delta R}$ | Paired Bootstrap $95\%$ CI | Statistical Significance |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **A0** | Reference Architecture | $0.289900$ | $0.289900$ | $0.000000$ | — | Reference |
| **A1** | No Confidence ($\alpha=0$) | $0.289900$ | $0.274761$ | $-0.015139$ | $[-0.018219, -0.011936]$ | Significant ($p < 0.001$) |
| **A2** | No Reliability ($\beta=0$) | $0.289900$ | $0.283707$ | $-0.006193$ | $[-0.006796, -0.005625]$ | Significant ($p < 0.001$) |
| **A3** | No Uncertainty ($\gamma=0$) | $0.289900$ | $0.320742$ | $+0.030842$ | $[+0.027805, +0.033985]$ | Significant ($p < 0.001$) |
| **A4** | No Quality ($\eta=0$) | $0.289900$ | $0.289308$ | $-0.000592$ | $[-0.000846, -0.000360]$ | Significant ($p < 0.001$) |
| **A5** | No DCRI Discount ($\delta=0$) | $0.289900$ | $0.289900$ | $+0.000000$ | $[+0.000000, +0.000000]$ | Invariant ($p = 1.0$) |
| **A_UNIFORM** | Uniform Baseline | $0.289900$ | $0.296804$ | $+0.006904$ | $[+0.001625, +0.012249]$ | Significant ($p = 0.008$) |

---

## 3. Analysis of Risk Propagation Drivers

### 3.1 Variant A3: Strongest Risk Inflation ($+0.030842$)
Eliminating the router uncertainty penalty causes the largest positive shift in fused risk across all variants:
- **Underlying Dynamics:** The foot ulcer model is trained on severe diabetic lesion pathology and tends to predict higher risk scores ($\overline{r_{\text{foot}}} = 0.384$) than the retinal screening model ($\overline{r_{\text{retina}}} = 0.246$) or clinical tabular model ($\overline{r_{\text{clinical}}} = 0.278$).
- In $A0$, the foot ulcer channel is penalized due to high epistemic uncertainty ($\overline{U_{\text{foot}}} \approx 0.42$).
- In $A3$, removing $-\gamma U_i$ unleashes foot ulcer weight by $+0.116634$. Because $r_{\text{foot}}$ is higher than the cohort average, this substantial weight reallocation directly inflates fused risk by $+0.030842$.

### 3.2 Variant A1: Risk Reduction from Tabular Reallocation ($-0.015139$)
Removing confidence produces a notable decrease in fused risk:
- **Underlying Dynamics:** In $A0$, high confidence spikes are most frequent on retina images with detected pathology. In $A1$, setting $\alpha=0$ shifts $+0.098873$ weight into clinical tabular data, which tends to predict lower, more moderate risk scores on these encounters.
- As a consequence, fused risk decreases by $-0.015139$.

### 3.3 Variant A2: Modest Risk Moderation ($-0.006193$)
Removing reliability priors moderately deflates fused risk:
- **Underlying Dynamics:** $A0$ weights vision models higher than tabular models ($0.85, 0.80$ vs $0.75$). In $A2$, leveling the reliability prior transfers weight toward tabular data ($\Delta w_{\text{clinical}} = +0.028372$), generating a modest downward drift in fused risk.

### 3.4 Variant A4: Negligible Shift ($-0.000592$)
The quality bonus produces negligible fused risk displacement:
- The $95\%$ bootstrap CI $[-0.000846, -0.000360]$ excludes zero, confirming statistical detection, but the practical effect size is near zero ($< 0.001$).

### 3.5 Variant A5: Invariant Decoupling
Because A5 only modifies the downstream DCRI formulation, fused risk propagation is exactly zero for all 500 packets:
$$\forall k \in \{1, \dots, 500\}, \quad R_{\text{fusion}}^{(A5)}(k) - R_{\text{fusion}}^{(A0)}(k) = 0.000000$$

This establishes the formal decoupling between router weight allocation and downstream DCRI risk calculation.

### 3.6 Relationship to Downstream DCRI Shifts
Because the canonical DCRI epistemic uncertainty penalty:
$$P_U = \delta^* \sum_{i \in \mathcal{A}} U_i$$
is an unweighted sum over active channels $\mathcal{A}$ (established in Phase C11.6 and sealed in Phase C11.13/C11.15), the penalty is invariant to routing weight allocations ($\Delta P_U = 0.0$). Consequently, for all router-level ablations ($A1$, $A2$, $A3$, $A4$, and $A_{\text{UNIFORM}}$), the paired DCRI shift is mathematically identical to the fused risk shift:
$$\Delta \text{DCRI}^{(v)} = \Delta R_{\text{fusion}}^{(v)} - \Delta P_U = \Delta R_{\text{fusion}}^{(v)} - 0.0 \equiv \Delta R_{\text{fusion}}^{(v)}$$
This explains why the mean paired differences and confidence intervals for $\Delta R_{\text{fusion}}$ and $\Delta \text{DCRI}$ match exactly across $A1$–$A4$ and $A_{\text{UNIFORM}}$, while diverging uniquely in $A5$ where the discount parameter $\delta$ itself is ablated.
