# Chapter 01 — Scientific Protocol & Research Hypotheses

## 1. Scientific Context & Purpose

In the FusionMedAI framework, the Decision Confidence & Risk Index ($\text{DCRI}$) provides a derived, uncertainty-discounted decision-support index computed downstream of dynamic ACARA-U routing:

$$\text{DCRI}_\delta = R_{\text{fusion}} - \delta \sum_{i \in \mathcal{A}} U_i$$

where:
- $R_{\text{fusion}} = \sum_{i \in \mathcal{A}} w_i r_i \in [0.0, 1.0]$ is the weighted decision-level risk aggregation under frozen reference routing weights $w_i$.
- $U_i \in [0.0, 1.0]$ is the predictive uncertainty for modality channel $i$.
- $\mathcal{A} \subseteq \{\text{retina}, \text{foot}, \text{clinical}\}$ represents the active modality set.
- $\delta \in [0.0, 1.0]$ is the global scalar penalty multiplier governing how aggressively aggregate epistemic uncertainty discounts the base fused risk.

Historically, $\delta = 0.20$ was used as a provisional convenience point during Phase C11.6 development. Phase C11.13 formally investigates the candidate parameter space $\delta \in [0.0, 1.0]$ to establish an empirically grounded parameter freeze.

---

## 2. Research Objective

Phase C11.13 addresses exactly one core research question:

> **Primary Research Question:**  
> What value of the DCRI uncertainty penalty multiplier $\delta \in [0.0, 1.0]$ should be selected over the frozen controlled decision cohort ($N=500$, seed 115) to achieve meaningful uncertainty attenuation while preserving index interpretability, modality regime stability, and rank fidelity, without post-hoc test-set label optimization?

---

## 3. Strict Methodological Exclusions

To prevent methodological corruption, overfitting, and ungrounded claims, Phase C11.13 strictly adheres to the following boundaries:

1. **Zero Ground-Truth Label Optimization**: $\delta$ is NOT selected by maximizing ROC-AUC, PR-AUC, or Brier score against any patient ground truth. DCRI is a decision-support index, not a calibrated posterior probability.
2. **No Post-Hoc Grid Refinement**: Candidate points are pre-specified before execution. Searching intermediate fractional points (e.g. $0.11, 0.12$) after observing results is explicitly prohibited.
3. **No Upstream Perturbations**:
   - Router coefficients remain frozen at $\Theta_0 = (\alpha=1.0, \beta=1.5, \gamma=1.0, \eta=0.5)$.
   - Modality risk projections $r_i$, reliabilities $R_i$, qualities $Q_i$, availabilities $A_i$, and uncertainties $U_i$ are bitwise identical to C11 frozen inputs.
4. **No Clamping of Negative DCRI**: Negative DCRI values ($R_{\text{fusion}} < \delta \sum U_i$) reflect high epistemic uncertainty relative to risk and are strictly preserved.
5. **Precise Cohort Provenance**: Packets are controlled multimodal decision evaluation units assembled from validated modality model outputs (Retina, Foot, Tabular Clinical) under frozen benchmark generation (seed 115, $N=500$); they are benchmark units, not longitudinally paired real-patient encounters or prospective clinical trial records.

---

## 4. Pre-Specified Hypotheses

The following formal hypotheses were registered prior to experimental execution:

| Hypothesis ID | Statement | Verification Metric |
| :--- | :--- | :--- |
| **H1 (Zero Identity)** | At $\delta=0.0$, $\text{DCRI}_0 \equiv R_{\text{fusion}}$ bitwise for all packets. | $\max \lvert \text{DCRI}_0 - R_{\text{fusion}} \rvert < 10^{-14}$ |
| **H2 (Monotonic Decay)** | For any packet $k$ and $\delta_2 > \delta_1$, $\text{DCRI}_{\delta_2, k} \le \text{DCRI}_{\delta_1, k}$. | $\text{DCRI}_{\delta_2} - \text{DCRI}_{\delta_1} \le 0$ across all $N=500$ |
| **H3 (Linear Sensitivity)** | Cohort mean $\overline{\text{DCRI}}$ decreases linearly with slope exactly $-\overline{U_{\text{sum}}}$. | $\lvert \frac{\Delta \overline{\text{DCRI}}}{\Delta \delta} - (-\overline{U_{\text{sum}}}) \rvert < 10^{-12}$ |
| **H4 (Negative Emergence)** | Moderate-to-high penalty ($\delta \ge 0.10$) produces non-zero negative DCRI rates. | $P(\text{DCRI}_\delta < 0) > 0$ for $\delta \ge 0.10$ |
| **H5 (Rank Divergence)** | Spearman correlation $\rho_s(\text{DCRI}_\delta, R_{\text{fusion}})$ monotonically decays with $\delta$. | $\rho_s(\delta_{m+1}) \le \rho_s(\delta_m)$ |
| **H6 (Regime Monotonicity)** | Modality cardinality $M = \lvert \mathcal{A} \rvert$ increases mean uncertainty burden $\overline{U_{\text{sum}}}$, accelerating negative rates. | $P(\text{DCRI} < 0 \mid M=3) > P(\text{DCRI} < 0 \mid M=1)$ |

> [!NOTE]
> **Hypothesis vs Finding Distinction**: Hypotheses H1–H6 represent pre-specified expected mathematical and statistical properties. Empirical verification and numerical confirmation of these hypotheses are detailed in Chapter 03, Chapter 04, and verified in Chapter 07.
