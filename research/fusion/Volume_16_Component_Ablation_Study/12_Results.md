# Chapter 12: Experimental Results

## 1. Executive Experimental Scorecard

Phase C11.16 systematically evaluated 8 model configurations across $N=500$ controlled decision packets under natural availability masks and across 8 availability regimes. The grand total of $28,500$ packet evaluations is obtained as:
- **Valid Configurations:** $7$ valid candidate configurations ($A0$, $A1$, $A2$, $A3$, $A4$, $A5$, $A_{\text{UNIFORM}}$) $\times$ $4,000$ regime evaluations ($8$ availability regimes $\times$ $500$ packets) = $28,000$ evaluations.
- **Negative Control Candidate ($A6$):** $1$ targeted fault-injection candidate evaluated across $500$ packets under the bi-modal $RF$ regime = $500$ evaluations.
- **Total:** $28,000 + 500 = 28,500$ total packet evaluations.

The table below presents the consolidated results matrix:

| Variant | Target | $\overline{w_{\text{ret}}}$ | $\overline{w_{\text{foot}}}$ | $\overline{w_{\text{clin}}}$ | $\overline{R}$ | $\overline{\text{DCRI}}$ | $\Delta \overline{R}$ vs A0 | $\Delta \overline{\text{DCRI}}$ vs A0 | Reclassified vs A0 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A0** | Reference ($\Theta_0$) | $0.4271$ | $0.2231$ | $0.3498$ | $0.2610$ | $0.1976$ | $0.0000$ | $0.0000$ | $0$ ($0.0\%$) |
| **A1** | No Confidence ($\alpha=0$) | $0.4301$ | $0.2330$ | $0.3369$ | $0.2748$ | $0.2114$ | $+0.0137$ | $+0.0137$ | $34$ ($6.8\%$) |
| **A2** | No Reliability ($\beta=0$) | $0.4026$ | $0.2128$ | $0.3846$ | $0.2532$ | $0.1898$ | $-0.0078$ | $-0.0078$ | $21$ ($4.2\%$) |
| **A3** | No Uncertainty ($\gamma=0$) | $0.3642$ | $0.3284$ | $0.3074$ | $0.2914$ | $0.2280$ | $+0.0303$ | $+0.0303$ | $63$ ($12.6\%$) |
| **A4** | No Quality ($\eta=0$) | $0.4175$ | $0.2234$ | $0.3591$ | $0.2598$ | $0.1965$ | $-0.0012$ | $-0.0012$ | $2$ ($0.4\%$) |
| **A5** | No Discount ($\delta=0$) | $0.4271$ | $0.2231$ | $0.3498$ | $0.2610$ | $0.2610$ | $0.0000$ | $+0.0634$ | $96$ ($19.2\%$) |
| **A_UNIFORM** | Uniform | $0.3333$ | $0.3333$ | $0.3333$ | $0.2968$ | $0.2334$ | $+0.0358$ | $+0.0358$ | $89$ ($17.8\%$) |
| **A6** | Negative Control | — | — | — | — | — | — | — | **REJECTED (500/500)** |

*Note: In natural reference availability encounters, mean weights for tri-modal encounters are reported above.*

---

## 2. Key Scientific Observations

### 2.1 The Critical Role of Epistemic Uncertainty Penalization
The predictive uncertainty penalty $-\gamma U_i$ serves as the foundational safeguard preventing the router from assigning excessive decision authority to uncalibrated or high-variance modalities.
- When $\gamma$ is removed ($A3$), foot ulcer weight surges by $+10.5\%$ ($0.2231 \to 0.3284$), driving a $+0.0303$ inflation in fused risk ($0.2610 \to 0.2914$) and reclassifying $63$ encounters across decision tiers.
- In contrast, downstream discounting ($\delta=0.10$) reduces score levels globally by $0.0634$, serving as a downstream buffer against overconfidence.

### 2.2 Encounter Confidence and Historical Reliability Priors
Both encounter-specific confidence ($\alpha C_i$) and static reliability priors ($\beta R_i$) actively shape multi-modality weight allocation:
- Removing confidence ($A1, \alpha=0$) shifts mean weights to $(0.4301, 0.2330, 0.3369)$, increasing mean fused risk by $+0.0137$ ($0.2610 \to 0.2748$) and reclassifying $34$ packets.
- Removing reliability ($A2, \beta=0$) shifts mean weights to $(0.4026, 0.2128, 0.3846)$, decreasing mean fused risk by $-0.0078$ ($0.2610 \to 0.2532$) and reclassifying $21$ packets.

### 2.3 Quality Bonus Acts as a Mild Regularizer
Under standard benchmark conditions, the quality bonus ($\eta Q_i$) produces minor routing adjustment ($\Delta R = -0.0012$, $2$ packets reclassified). Its primary utility occurs under targeted input degradation scenarios (as demonstrated in Volume 10).

### 2.4 Downstream DCRI Discounting Is Purely Orthogonal
Ablating $\delta$ ($A5$) has identically zero impact on routing weights and raw fused risk ($\Delta w \equiv 0.0, \Delta R \equiv 0.0$). Its effect is entirely confined to score level translation ($+0.0634$) and threshold crossing ($96$ packets reclassified across tiers).

---

## 3. Methodological Boundary & Constraints

- **Controlled Packets, Not Clinical Trials:** These results quantify the mathematical behavior of the ACARA-U routing formulation over controlled multimodal decision packets. They do not constitute clinical validation on patient-paired clinical cohorts.
- **Fixed Frozen Reference:** No parameters were retuned. The study exclusively establishes component attribution relative to the frozen reference configuration $\Theta_0 = (1.0, 1.5, 1.0, 0.5, 0.10)$.
