# Chapter 01 — Scientific Protocol & Research Hypotheses

## 1. Research Protocol Specification

### Title
**Phase C11.14: Decision-Policy Sensitivity and Operating-Behavior Analysis of the Frozen DCRI Index**

### Core Research Question
> **How sensitive are hypothetical decision-policy outputs to pre-specified DCRI thresholds and modality availability conditions when the uncertainty penalty multiplier is fixed at $\delta^* = 0.10$?**

---

## 2. Parameter & Upstream State Freezing

Phase C11.14 strictly evaluates decision policy behavior over a frozen pipeline. No parameter tuning, hyperparameter search, or model retraining is permitted.

| Pipeline Component | Specification | Frozen Provenance |
| :--- | :--- | :--- |
| **Uncertainty Penalty Multiplier** | $\delta^* = 0.10$ (`D10`) | Selected and sealed in Phase C11.13 |
| **Router Logit Kernel** | $z_i = 1.0 C_i + 1.5 R_i - 1.0 U_i + 0.5 Q_i$ | $\Theta_0$ reference configuration sealed in C11.12 |
| **Prior Modality Reliabilities** | $R_R = 0.929956, R_F = 0.922266, R_C = 0.825382$ | Sealed validation priors from C11.3 |
| **Evaluation Cohort** | $N=500$ controlled decision packets (`seed=115`) | Fixed benchmark sequence `PACKET_0000` to `PACKET_0499` |
| **Operational Thresholds** | Standard: $\tau_1 = 0.20, \tau_2 = 0.40$; Grid: $\tau_1 \in [0.10, 0.30], \tau_2 \in [0.35, 0.60]$ | Pre-specified prior to evaluation |

---

## 3. Pre-Specified Hypotheses

- **Hypothesis H1 (Monotonic Action Non-Inflation)**: Because $\text{DCRI}_{0.10} = R_{\text{fusion}} - 0.10 \sum U_i \le R_{\text{fusion}}$ pointwise for every packet ($\sum U_i \ge 0$), Policy A ($\text{DCRI}_{0.10}$) will never upgrade any packet to a higher action tier relative to Policy B ($R_{\text{fusion}}$ reference). The upward reclassification rate will be exactly $0.0\%$.
- **Hypothesis H2 (Workload & Escalation Attenuation)**: The introduction of uncertainty discounting will reduce the overall proportion of packets assigned to the highest action tier (Escalation), shifting ambiguous or high-uncertainty encounters toward intermediate monitoring or routine review.
- **Hypothesis H3 (Modality Composition & Aggregate Uncertainty Dependency)**: Reclassification magnitude will depend on the specific modality composition and resulting aggregate predictive uncertainty ($\sum U_i$) of the active channels in this benchmark, with high-uncertainty regimes experiencing greater action adjustments than low-uncertainty regimes, irrespective of modality count alone.
- **Hypothesis H4 (Boundary Stability Under Jitter)**: Small continuous perturbations of the threshold boundaries ($\pm 0.02$) will produce bounded, monotonic shifts in action allocation without threshold bifurcation or instability.

---

## 4. Methodological Scope & Boundaries

1. **Simulated Workflow Only**: Action categories (Routine Review, Additional Assessment, Escalation for Human Review) are hypothetical operational routing labels to benchmark threshold mechanics. They do not represent validated clinical pathways or patient risk categories.
2. **Absence of Clinical Ground Truth**: The controlled benchmark cohort ($N=500$) does not contain paired patient-level multimodal clinical outcomes. No clinical utility, net benefit curves, or diagnostic accuracy claims are derived.
3. **No Retuning of $\delta$**: $\delta^* = 0.10$ is locked. This phase evaluates operating behavior, not a renewed search for a "better" $\delta$.
