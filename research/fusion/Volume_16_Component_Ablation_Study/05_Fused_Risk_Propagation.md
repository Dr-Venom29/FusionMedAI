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
| **A0** | Reference Architecture | $0.261017$ | $0.261017$ | $0.000000$ | — | Reference |
| **A1** | No Confidence ($\alpha=0$) | $0.261017$ | $0.274761$ | $+0.013744$ | $[+0.011245, +0.016254]$ | Significant ($p < 0.001$) |
| **A2** | No Reliability ($\beta=0$) | $0.261017$ | $0.253192$ | $-0.007825$ | $[-0.008456, -0.007194]$ | Significant ($p < 0.001$) |
| **A3** | No Uncertainty ($\gamma=0$) | $0.261017$ | $0.291355$ | $+0.030338$ | $[+0.027210, +0.033466]$ | Significant ($p < 0.001$) |
| **A4** | No Quality ($\eta=0$) | $0.261017$ | $0.259849$ | $-0.001168$ | $[-0.001422, -0.000914]$ | Significant ($p < 0.001$) |
| **A5** | No DCRI Discount ($\delta=0$) | $0.261017$ | $0.261017$ | $+0.000000$ | $[+0.000000, +0.000000]$ | Invariant ($p = 1.0$) |
| **A_UNIFORM** | Uniform Baseline | $0.261017$ | $0.296804$ | $+0.035787$ | $[+0.030412, +0.041162]$ | Significant ($p < 0.001$) |

---

## 3. Analysis of Risk Propagation Drivers

### 3.1 Variant A3: Strongest Risk Inflation ($+0.030338$)
Eliminating the router uncertainty penalty causes the largest positive shift in fused risk across all variants:
- **Underlying Dynamics:** The foot ulcer model is trained on severe diabetic lesion pathology and tends to predict higher risk scores ($\overline{r_{\text{foot}}} = 0.4996$) than the retinal screening model ($\overline{r_{\text{retina}}} = 0.2714$) or clinical tabular model ($\overline{r_{\text{clinical}}} = 0.1194$).
- In $A0$, the foot ulcer channel is penalized due to high epistemic uncertainty ($\overline{U_{\text{foot}}} \approx 0.3712$).
- In $A3$, removing $-\gamma U_i$ unleashes foot ulcer weight from $0.223079 \to 0.328396$ ($+0.105317$). Because $r_{\text{foot}}$ is substantially higher than the cohort average, this weight reallocation directly inflates fused risk by $+0.030338$ ($0.261017 \to 0.291355$).

### 3.2 Variant A1: Risk Inflation from Confidence Ablation ($+0.013744$)
Removing dynamic confidence increases fused risk:
- **Underlying Dynamics:** In $A0$, high decision-boundary confidence ($C_i = \lvert p_1 - p_0 \rvert$) frequently stabilizes the low-risk clinical tabular channel ($r_{\text{clinical}} \approx 0.1194$). In $A1$, setting $\alpha=0$ removes confidence-based weighting, redistributing weight away from clinical tabular ($w_{\text{clinical}}$ drops from $0.3498$ to $0.3369$) and boosting higher-risk channels ($w_{\text{retina}}$ rises to $0.4301$, $w_{\text{foot}}$ rises to $0.2330$), increasing mean fused risk by $+0.013744$ ($0.261017 \to 0.274761$).

### 3.3 Variant A2: Modest Risk Moderation ($-0.007825$)
Removing reliability priors moderately deflates fused risk:
- **Underlying Dynamics:** $A0$ assigns higher frozen reliability priors to vision channels ($R_R = 0.929956, R_F = 0.922266$) than clinical tabular ($R_C = 0.825382$). In $A2$, removing $\beta R_i$ transfers weight toward clinical tabular data ($\Delta w_{\text{clinical}} = +0.0348$), generating a modest downward drift in fused risk ($0.261017 \to 0.253192$).

### 3.4 Variant A4: Negligible Shift ($-0.001168$)
The quality bonus produces minor fused risk displacement:
- The paired difference is small ($\Delta R = -0.001168$, $95\%$ bootstrap CI $[-0.001422, -0.000914]$), confirming statistical detection with minimal practical effect size under standard benchmark inputs.

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
