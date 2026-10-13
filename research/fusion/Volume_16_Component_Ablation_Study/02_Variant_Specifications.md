# Chapter 02: Variant Specifications

## 1. Parameter Specifications

The ablation suite specifies 8 concrete configurations. All non-ablated coefficients retain their frozen reference values:
- Reference Confidence Weight: $\alpha = 1.0$
- Reference Reliability Weight: $\beta = 1.5$
- Reference Uncertainty Penalty: $\gamma = 1.0$
- Reference Quality Bonus: $\eta = 0.5$
- Reference DCRI Discount: $\delta^* = 0.10$

| Variant ID | Name | $\alpha$ | $\beta$ | $\gamma$ | $\eta$ | $\delta$ | Masking Protocol |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **A0** | Full B6 Reference | $1.0$ | $1.5$ | $1.0$ | $0.5$ | $0.10$ | Active set masking |
| **A1** | No Confidence | $0.0$ | $1.5$ | $1.0$ | $0.5$ | $0.10$ | Active set masking |
| **A2** | No Reliability Prior | $1.0$ | $0.0$ | $1.0$ | $0.5$ | $0.10$ | Active set masking |
| **A3** | No Uncertainty Penalty | $1.0$ | $1.5$ | $0.0$ | $0.5$ | $0.10$ | Active set masking |
| **A4** | No Quality Bonus | $1.0$ | $1.5$ | $1.0$ | $0.0$ | $0.10$ | Active set masking |
| **A5** | No DCRI Uncertainty Discount | $1.0$ | $1.5$ | $1.0$ | $0.5$ | $0.00$ | Active set masking |
| **A_UNIFORM** | Uniform Active Baseline | $0.0$ | $0.0$ | $0.0$ | $0.0$ | $0.10$ | Equal weighting over $\mathcal{A}$ |
| **A6** | Negative Control Fault Injection | $1.0$ | $1.5$ | $1.0$ | $0.5$ | $0.10$ | Inactive channel forced $\ge 0.10$ |

---

## 2. Mathematical Variant Formulations

### 2.1 Variant A0: Full B6 Reference
Logit formulation for active channels $i \in \mathcal{A}$:
$$z_i^{(A0)} = 1.0 C_i + 1.5 R_i - 1.0 U_i + 0.5 Q_i$$

$$w_i^{(A0)} = \frac{\exp(z_i^{(A0)})}{\sum_{j \in \mathcal{A}} \exp(z_j^{(A0)})}$$

$$\mathrm{DCRI}^{(A0)} = \sum_{i \in \mathcal{A}} w_i^{(A0)} r_i - 0.10 \sum_{i \in \mathcal{A}} U_i$$

### 2.2 Variant A1: No Confidence Contribution
Removes dynamic modality prediction confidence:
$$z_i^{(A1)} = 1.5 R_i - 1.0 U_i + 0.5 Q_i$$
Evaluates the degree to which encounter-specific model confidence drives weight reallocation relative to static reliability priors.

### 2.3 Variant A2: No Reliability Contribution
Removes global historical reliability priors ($R_{\text{retina}}=0.929956, R_{\text{foot}}=0.922266, R_{\text{clinical}}=0.825382$):
$$z_i^{(A2)} = 1.0 C_i - 1.0 U_i + 0.5 Q_i$$
Tests whether the router relies excessively on pre-computed validation accuracy versus dynamic case-level signals.


### 2.4 Variant A3: No Router Uncertainty Penalty
Eliminates the predictive uncertainty penalty $-\gamma U_i$ from router logit generation:
$$z_i^{(A3)} = 1.0 C_i + 1.5 R_i + 0.5 Q_i$$
Crucially, downstream DCRI uncertainty discounting ($\delta^* = 0.10$) remains active. This decouples router-level uncertainty avoidance from downstream score conservatism.

### 2.5 Variant A4: No Input Quality Bonus
Removes the input quality term $+\eta Q_i$:
$$z_i^{(A4)} = 1.0 C_i + 1.5 R_i - 1.0 U_i$$
Quantifies the impact of signal degradation bonuses on decision weight allocation.

### 2.6 Variant A5: No Downstream DCRI Uncertainty Discount
Preserves reference router logits ($z_i^{(A5)} \equiv z_i^{(A0)}$), producing identical decision weights ($w_i^{(A5)} \equiv w_i^{(A0)}$) and fused risk ($R_{\text{fusion}}^{(A5)} \equiv R_{\text{fusion}}^{(A0)}$), but disables the downstream discount:
$$\delta^{(A5)} = 0.0 \implies P_U^{(A5)} = 0.0$$
$$\text{DCRI}^{(A5)} = R_{\text{fusion}}^{(A5)}$$
Directly quantifies the exact shift introduced by downstream epistemic conservatism.

### 2.7 Diagnostic Baseline A_UNIFORM: Uniform Active Normalization
Sets all routing coefficients to zero ($\alpha=\beta=\gamma=\eta=0$):
$$z_i^{(\text{UNIFORM})} = 0.0 \quad \forall i \in \mathcal{A}$$
$$w_i^{(\text{UNIFORM})} = \frac{1}{\lvert \mathcal{A} \rvert} \quad \text{for } i \in \mathcal{A}, \quad w_j = 0 \quad \text{for } j \notin \mathcal{A}$$
Serves as an empirical baseline to measure the total value of dynamic weighting over naive averaging.

### 2.8 Variant A6: Negative Control (Deliberate Mask Violation)
Computes reference logits, but injects a deliberate violation into availability masking:
$$\text{If } \lvert \mathcal{A} \rvert < 3 \text{ and } \exists j \notin \mathcal{A}, \text{ set } w_j = 0.10 \text{ and renormalize } w_{i \in \mathcal{A}} \text{ to } 0.90.$$
This variant must be immediately flagged, rejected, and denied certification by Gate C16-04.
