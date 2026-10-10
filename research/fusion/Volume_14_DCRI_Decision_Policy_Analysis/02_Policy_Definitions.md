# Chapter 02 — Policy Definitions & Threshold Framework

## 1. Operational Action Taxonomy

To benchmark how decision-level uncertainty discounting alters hypothetical downstream routing, we define a three-tiered hypothetical action taxonomy:

| Action Tier | Tier Identifier | Operational Role (Hypothetical Routing Label) | Assignment Condition |
| :---: | :---: | :--- | :--- |
| **Tier 0** | `ROUTINE_REVIEW` | Lower-tier routine review / standard monitoring queue | $\text{Score} < \tau_1$ |
| **Tier 1** | `ADDITIONAL_ASSESSMENT` | Intermediate-tier additional assessment / secondary inspection | $\tau_1 \le \text{Score} < \tau_2$ |
| **Tier 2** | `ESCALATION` | Higher-tier escalation / specialist review queue | $\text{Score} \ge \tau_2$ |

> [!NOTE]
> **Hypothetical Routing Labels**: These tier names are operational placeholders designed strictly to test decision-rule mechanics on synthetic decision packets. They do not represent clinically validated low-, intermediate-, or high-risk disease categories, nor do they imply safe clinical discharge thresholds without prospectively validated clinical trials.

---

## 2. Policy Family Formulations

We compare two explicit policy families applied to the frozen cohort:

### Policy A: DCRI Policy ($\delta^* = 0.10$)
Assigns actions based on the uncertainty-discounted decision index:
$$\text{Action}_{\text{DCRI}} = f(\text{DCRI}_{0.10}; \tau_1, \tau_2), \quad \text{where } \text{DCRI}_{0.10} = R_{\text{fusion}} - 0.10 \sum_{i \in \mathcal{A}} U_i$$

- **Negative DCRI Handling**: Negative values ($\text{DCRI} < 0$) satisfy $\text{DCRI} < \tau_1$ for any valid $\tau_1 > 0$, mapping strictly into **Tier 0 (Routine Review)**. In this mathematical model, negative values reflect instances where aggregate uncertainty penalty exceeds the initial fused risk estimate.

### Policy B: Fused-Risk Reference Policy (Unpenalized)
Assigns actions based directly on the unpenalized weighted fused risk:
$$\text{Action}_{\text{Fused}} = f(R_{\text{fusion}}; \tau_1, \tau_2), \quad \text{where } R_{\text{fusion}} = \sum_{i \in \mathcal{A}} w_i r_i$$

---

## 3. Threshold Specifications

### Standard Operating Point (Nominal Baseline)
- **Lower Operating Threshold ($\tau_1$)**: $0.20$
- **Upper Operating Threshold / Escalation Boundary ($\tau_2$)**: $0.40$

### Systematic Sensitivity Grid (25 Valid Pairs)
To evaluate operating sensitivity across different parameterizations, a full 2-D grid is evaluated across all $(\tau_1, \tau_2)$ such that $\tau_1 < \tau_2$:
- $\tau_1 \in \{0.10, 0.15, 0.20, 0.25, 0.30\}$
- $\tau_2 \in \{0.35, 0.40, 0.45, 0.50, 0.60\}$
