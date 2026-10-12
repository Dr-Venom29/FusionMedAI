# Chapter 12: Experimental Results

## 1. Executive Experimental Scorecard

Phase C11.16 systematically evaluated 8 model configurations across $N=500$ controlled decision packets under natural availability masks and across 8 availability regimes. The grand total of $28,500$ packet evaluations is obtained as:
- **Valid Configurations:** $7$ valid candidate configurations ($A0$, $A1$, $A2$, $A3$, $A4$, $A5$, $A_{\text{UNIFORM}}$) $\times$ $4,000$ regime evaluations ($8$ availability regimes $\times$ $500$ packets) = $28,000$ evaluations.
- **Negative Control Candidate ($A6$):** $1$ targeted fault-injection candidate evaluated across $500$ packets under the bi-modal $RF$ regime = $500$ evaluations.
- **Total:** $28,000 + 500 = 28,500$ total packet evaluations.

The table below presents the consolidated results matrix:

| Variant | Target | $\overline{w_{\text{ret}}}$ | $\overline{w_{\text{foot}}}$ | $\overline{w_{\text{clin}}}$ | $\overline{R}$ | $\overline{\text{DCRI}}$ | Neg DCRI | Tier 0 (Routine) | Tier 1 (Additional) | Tier 2 (Escalation) | Net Escalation |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A0** | Reference | $0.5010$ | $0.2609$ | $0.2381$ | $0.2899$ | $0.2265$ | $37$ | $232$ | $187$ | $81$ | — |
| **A1** | No Confidence | $0.4301$ | $0.2330$ | $0.3369$ | $0.2748$ | $0.2114$ | $44$ | $241$ | $201$ | $58$ | $-23$ ($-28.4\%$) |
| **A2** | No Reliability | $0.4805$ | $0.2531$ | $0.2664$ | $0.2837$ | $0.2203$ | $38$ | $237$ | $190$ | $73$ | $-8$ ($-9.9\%$) |
| **A3** | No Uncertainty | $0.4180$ | $0.3776$ | $0.2045$ | $0.3207$ | $0.2573$ | $29$ | $196$ | $208$ | $96$ | $+15$ ($+18.5\%$) |
| **A4** | No Quality | $0.4920$ | $0.2625$ | $0.2455$ | $0.2893$ | $0.2259$ | $37$ | $231$ | $188$ | $81$ | $0$ ($0.0\%$) |
| **A5** | No Discount | $0.5010$ | $0.2609$ | $0.2381$ | $0.2899$ | $0.2899$ | $0$ | $176$ | $207$ | $117$ | $+36$ ($+44.4\%$) |
| **A_UNIFORM** | Uniform | $0.3333$ | $0.3333$ | $0.3333$ | $0.2968$ | $0.2334$ | $34$ | $208$ | $240$ | $52$ | $-29$ ($-35.8\%$) |
| **A6** | Negative Control | — | — | — | — | — | — | — | — | — | **REJECTED (500/500)** |

*Note: In natural reference availability encounters, mean weights for tri-modal encounters are reported above.*

---

## 2. Key Scientific Observations

### 2.1 The Critical Role of Epistemic Uncertainty Penalization
The predictive uncertainty penalty $-\gamma U_i$ serves as the foundational safeguard preventing the router from assigning excessive decision authority to uncalibrated or high-variance modalities.
- When $\gamma$ is removed ($A3$), foot ulcer weight surges by $+11.7\%$, driving a $+0.0308$ inflation in fused risk and escalating $15$ additional encounters into Tier 2 (Escalation).
- In contrast, downstream discounting ($\delta=0.10$) reduces score levels globally by $0.0634$, serving as a downstream buffer against overconfidence.

### 2.2 Dynamic Confidence Outweighs Historical Priors
The contribution of encounter-specific confidence ($\alpha C_i$) exceeds that of static reliability priors ($\beta R_i$) by a factor of $3.5\times$ in routing weight shift magnitude ($0.1977$ vs $0.0567$).
- Removing confidence ($A1$) suppresses vision modalities, shifting almost $10\%$ of total decision weight into tabular clinical data and reducing acute escalations by $28.4\%$.

### 2.3 Quality Bonus Acts as a Mild Regularizer
Under standard benchmark conditions, the quality bonus ($\eta Q_i$) produces negligible routing disruption ($\sum |\Delta w| = 0.0180$, $\Delta R = -0.0006$). Its primary utility occurs under targeted input degradation scenarios (as demonstrated in Volume 10).

### 2.4 Downstream DCRI Discounting Is Purely Orthogonal
Ablating $\delta$ ($A5$) has identically zero impact on routing weights and raw fused risk ($\Delta w \equiv 0.0, \Delta R \equiv 0.0$). Its effect is entirely confined to score level translation and threshold crossing.

---

## 3. Methodological Boundary & Constraints

- **Controlled Packets, Not Clinical Trials:** These results quantify the mathematical behavior of the ACARA-U routing formulation over controlled multimodal decision packets. They do not constitute clinical validation on patient-paired clinical cohorts.
- **Fixed Frozen Reference:** No parameters were retuned. The study exclusively establishes component attribution relative to the frozen reference configuration $\Theta_0 = (1.0, 1.5, 1.0, 0.5, 0.10)$.
