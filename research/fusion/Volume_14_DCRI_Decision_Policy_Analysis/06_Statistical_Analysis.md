# Chapter 06 — Statistical Analysis & Confidence Bounds

## 1. Paired Resampling Statistics ($B=1000$, Seed 115)

To quantify the statistical precision of observed policy shifts over the frozen $N=500$ controlled packet cohort, $B=1000$ paired bootstrap iterations were evaluated alongside Wilson score analytical intervals:

| Metric Dimension | Point Estimate | Bootstrap Mean | Bootstrap Std Error | 95% Bootstrap Percentile CI | 95% Wilson Score CI |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Score Discount ($\Delta = R_{\mathrm{fusion}} - \mathrm{DCRI}_{0.10}$)** | $+0.063394$ | $+0.063379$ | $0.001091$ | $[+0.061257, +0.065457]$ | — |
| **Total Reclassification Rate ($P(\text{Action}_A \ne \text{Action}_B)$)** | $18.40\%$ ($92/500$) | $18.36\%$ | $0.016922$ | $[15.20\%, 21.81\%]$ | $[15.25\%, 22.03\%]$ |
| **Monotonic Downgrade Rate** | $18.40\%$ ($92/500$) | $18.36\%$ | $0.016922$ | $[15.20\%, 21.81\%]$ | $[15.25\%, 22.03\%]$ |
| **Monotonic Upgrade Rate** | **$0.00\%$ ($0/500$)** | **$0.00\%$** | **$0.000000$** | **$[0.00\%, 0.00\%]$** | **$[0.00\%, 0.74\%]$** |
| **Escalation Workload Reduction Rate** | $7.20\%$ ($36/500$) | $7.20\%$ | $0.011634$ | $[5.00\%, 9.60\%]$ | $[5.25\%, 9.81\%]$ |

---

## 2. Statistical Interpretation & Boundaries

> [!NOTE]
> **Inferential Scope**:
> 1. **Non-Zero Separation**: The $95\%$ bootstrap confidence interval for the score discount strictly excludes zero ($[+0.061257, +0.065457] > 0$), establishing that the uncertainty discount produces a strictly non-degenerate shift across the evaluated packet sample.
> 2. **Reclassification Bounds**: The observed reclassification of $18.40\%$ of controlled packets at the nominal thresholds has a $95\%$ bootstrap percentile interval of $[15.20\%, 21.81\%]$ and a $95\%$ Wilson score interval of $[15.25\%, 22.03\%]$.
> 3. **Methodological Benchmark vs Clinical Population Bounds**: These confidence intervals characterize the internal sampling variability of the evaluated resampling procedure on the frozen $N=500$ benchmark dataset. They do not constitute confidence bounds on clinical efficacy, diagnostic sensitivity/specificity, or real-world safety in a live patient population.
