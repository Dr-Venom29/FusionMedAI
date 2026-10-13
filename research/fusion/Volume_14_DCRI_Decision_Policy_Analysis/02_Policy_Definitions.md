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

We compare three explicit policy configurations applied to the evaluated cohort:

### Policy A: Unconstrained DCRI Policy ($\delta^* = 0.10$)

Assigns actions based directly on the uncertainty-discounted decision index:

$$
\text{Action}_{\text{DCRI}} = f(\text{DCRI}_{0.10}; \tau_1, \tau_2), \quad \text{where } \text{DCRI}_{0.10} = R_{\text{fusion}} - 0.10 \sum_{i \in \mathcal{A}} U_i
$$

- **Negative DCRI Handling**: Negative values ($\text{DCRI} < 0$) satisfy $\text{DCRI} < \tau_1$ for any valid $\tau_1 > 0$, mapping strictly into **Tier 0 (Routine Review)**. In this mathematical model, negative values reflect instances where the aggregate uncertainty penalty exceeds the initial fused risk estimate.

### Policy B: Fused-Risk Reference Policy (Unpenalized)

Assigns actions based directly on the unpenalized weighted fused risk:

$$
\text{Action}_{\text{Fused}} = f(R_{\text{fusion}}; \tau_1, \tau_2), \quad \text{where } R_{\text{fusion}} = \sum_{i \in \mathcal{A}} w_i r_i
$$

### Policy S: Safeguarded DCRI Gateway Policy (Ambiguity & Repeat-Exam Protocol)

Decouples the numerical research index from the action tier assignment by imposing a safety gateway:

$$
\text{Action}_{\text{Safeguarded}} = \begin{cases}
\text{Action}_{\text{Fused}}, & \text{if } \text{Ambiguous}(\mathbf{x}) \text{ and } \text{Action}_{\text{DCRI}} < \text{Action}_{\text{Fused}} \\
\text{Action}_{\text{DCRI}}, & \text{otherwise}
\end{cases}
$$

where the ambiguity condition is defined over pre-declared experimental operating thresholds:

$$
\text{Ambiguous}(\mathbf{x}) \iff \left(\sum_{i \in \mathcal{A}} U_i \ge 0.30\right) \lor \left(\min_{i \in \mathcal{A}} Q_i < 0.60\right)
$$

- **Safeguard Invariant**: When evidence meets the predefined ambiguity criteria, Policy S prevents the DCRI-derived action tier from falling below the unpenalized fused-risk reference tier. When evidence does not meet those criteria, the DCRI-derived tier is retained. Ambiguous evidence also triggers a repeat-examination flag.
- **Flag Generation**: When $\text{Ambiguous}(\mathbf{x})$ is true, the gateway emits `REPEAT_EXAM_AMBIGUOUS_EVIDENCE` (defined by the constant `FLAG_REPEAT_EXAM_AMBIGUOUS_EVIDENCE`) along with audit reason codes (`HIGH_UNCERTAINTY`, `DEGRADED_QUALITY`, or `HIGH_UNCERTAINTY_AND_DEGRADED_QUALITY`).
- **Operating Threshold Status**: Thresholds $U_{\mathrm{sum}} = 0.30$ and $\min(Q) = 0.60$ are pre-declared experimental operating points selected to stress-test safety gating within the synthetic benchmark suite, not clinically validated diagnostic cutoffs.


---

## 3. Threshold Specifications

### Standard Operating Point (Nominal Baseline)
- **Lower Operating Threshold ($\tau_1$)**: $0.20$
- **Upper Operating Threshold / Escalation Boundary ($\tau_2$)**: $0.40$

### Systematic Sensitivity Grid (25 Valid Pairs)
To evaluate operating sensitivity across different parameterizations, a full 2-D grid is evaluated across all $(\tau_1, \tau_2)$ such that $\tau_1 < \tau_2$:
- $\tau_1 \in \{0.10, 0.15, 0.20, 0.25, 0.30\}$
- $\tau_2 \in \{0.35, 0.40, 0.45, 0.50, 0.60\}$
