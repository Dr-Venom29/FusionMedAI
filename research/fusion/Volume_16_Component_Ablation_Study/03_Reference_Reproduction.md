# Chapter 03: Reference B6 Reproduction

## 1. Reproduction Objective

Before evaluating component deletions, variant $A0$ must be rigorously demonstrated to reproduce the frozen Phase C11.13 and C11.15 baseline reference ($B6$) across the canonical $N=500$ decision packets (seed 115).

Numerical identity ensures that all observed paired shifts $\Delta = Av - A0$ represent pure component attribution rather than pipeline drifts, initialization differences, or numerical instability.

---

## 2. Upstream Reference Invariants

The reference baseline $B6$ established in Phase C11.13 and sealed in Phase C11.15 is characterized by the following frozen metrics over the 500-packet cohort:

| Parameter / Metric | Expected Frozen Value ($B6$) | Observed Value ($A0$) | Absolute Difference $\lvert \Delta \rvert$ | Verification Status |
| :--- | :---: | :---: | :---: | :---: |
| Cohort Size $N$ | $500$ | $500$ | $0$ | **EXACT MATCH** |
| Random Seed | $115$ | $115$ | $0$ | **EXACT MATCH** |
| Mean Fused Risk $\overline{R_{\text{fusion}}}$ | $0.261017...$ | $0.261017...$ | $< 10^{-15}$ | **EXACT MATCH** |
| Mean DCRI $\overline{\text{DCRI}}$ | $0.197624...$ | $0.197624...$ | $< 10^{-15}$ | **EXACT MATCH** |
| Mean Penalty $\overline{P_U}$ | $0.063394...$ | $0.063394...$ | $< 10^{-15}$ | **EXACT MATCH** |
| Negative DCRI Count | $38$ | $38$ | $0$ | **EXACT MATCH** |
| Simplex Violations | $0$ | $0$ | $0$ | **EXACT MATCH** |
| Mask Violations | $0$ | $0$ | $0$ | **EXACT MATCH** |

---

## 3. Reference Routing Weight Allocation

In the full 500-packet cohort under canonical reference router configuration $\Theta_0 = (1.0, 1.5, 1.0, 0.5)$, the reference architecture allocates weights across modalities as follows:

| Modality Channel | Mean Weight $\overline{w_i}$ | Std Dev $\sigma_w$ | Min Weight | Max Weight | Active Prior $R_i$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Retina Image** | $0.427135$ | $0.078806$ | $0.000000$ | $1.000000$ | $0.929956$ |
| **Foot Ulcer Image** | $0.223079$ | $0.077157$ | $0.000000$ | $1.000000$ | $0.922266$ |
| **Clinical Tabular** | $0.349785$ | $0.064987$ | $0.000000$ | $1.000000$ | $0.825382$ |
| **Simplex Sum $\sum w_i$** | $1.000000$ | $0.000000$ | $1.000000$ | $1.000000$ | — |

### Key Observations:
1. **Retina Dominance in Tri-Modal Settings:**
   Retina receives the largest mean weight ($0.427135$), driven jointly by its high historical reliability prior ($R_{\text{retina}}=0.929956$), high signal quality ($Q_{\text{retina}} \approx 0.77$), and low epistemic uncertainty ($U_{\text{retina}} \approx 0.165$).
2. **Foot Ulcer Suppression:**
   Foot ulcer receives the lowest mean weight ($0.223079$), primarily due to elevated epistemic uncertainty ($U_{\text{foot}} \approx 0.371$) triggering router-level penalty suppression via $-\gamma U_i$.
3. **Clinical Tabular Balance:**
   Clinical tabular retains a substantial baseline weight ($0.349785$), stabilized by its high feature completeness ($Q_{\text{clinical}} \approx 0.859$) and low predictive uncertainty ($U_{\text{clinical}} \approx 0.098$).

---

## 4. Reference Risk and Triage Distribution

The reference model produces the following distribution of scores across the $N=500$ packets:

| Metric | Mean | Std Dev | Median | 25th Pct | 75th Pct |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Fused Risk $R_{\text{fusion}}$** | $0.261017$ | $0.163198$ | $0.218151$ | $0.140210$ | $0.362140$ |
| **DCRI** | $0.197624$ | $0.172243$ | $0.150263$ | $0.076810$ | $0.298150$ |
| **Uncertainty Penalty $P_U$** | $0.063394$ | $0.014498$ | $0.063394$ | $0.052100$ | $0.074210$ |
| **Routing Entropy $H(w)$** | $1.038987$ | $0.040595$ | $1.046218$ | $1.012450$ | $1.069184$ |

### Hypothetical Triage Stratification:
- **Tier 0 (Routine Review, $\text{DCRI} < 0.20$):** $262$ packets ($52.4\%$)
- **Tier 1 (Additional Assessment, $0.20 \le \text{DCRI} < 0.40$):** $181$ packets ($36.2\%$)
- **Tier 2 (Escalation for Human Review, $\text{DCRI} \ge 0.40$):** $57$ packets ($11.4\%$)


The $A0$ implementation replicates all upstream benchmark metrics down to IEEE 754 double-precision tolerances ($< 10^{-15}$). Gate C16-02 is formally satisfied.
