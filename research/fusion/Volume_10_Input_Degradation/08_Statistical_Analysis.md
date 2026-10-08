# Statistical Rigor & Paired Bootstrap Analysis

## 1. Non-Parametric Paired Bootstrap Protocol

To account for inter-sample correlation in paired decision packets ($N=500$, $\text{seed}=115$), all confidence intervals are estimated via non-parametric paired bootstrapping with $B=1,000$ resamples:

$$
d_k = w_{i,k}^{(\text{degraded})} - w_{i,k}^{(\text{clean})}
$$

The empirical $95\%$ bootstrap confidence interval is derived from the $2.5^{\text{th}}$ and $97.5^{\text{th}}$ percentiles of the resampled mean distribution $\overline{d}^*$.

---

## 2. Statistical Outcomes Across Research Hypotheses

| Hypothesis | Evaluated Metric | Point Estimate (Severe D3) | $95\%$ Bootstrap CI | Statistical Outcome |
| :--- | :--- | :---: | :---: | :---: |
| **H1 (Quality Decay)** | $\Delta Q_R$ (Retina Blur) | $-0.6016$ | $[-0.6022, -0.6010]$ | **Supported** ($95\%$ CI strictly $< 0$) |
| **H2 (Authority Decay)** | $\Delta w_R$ (Retina Blur) | $-0.1697$ | $[-0.1710, -0.1684]$ | **Supported** ($95\%$ CI strictly $< 0$) |
| **H3 (Redistribution)** | $\sum_{j \ne R} \Delta w_j + \Delta w_R$ | $0.0000$ | $[0.0000, 0.0000]$ | **Confirmed** (Exact invariant) |
| **H4 (Positive Slope)** | $S_{QW}$ (Retina Blur) | $+0.2822$ | $[0.2799, 0.2842]$ | **Supported** ($95\%$ CI strictly $> 0$) |
| **H5 (Monotonicity)** | Monotonic Packets ($\%$) | $100.0\%$ | $[100.0\%, 100.0\%]$ | **Supported** ($>90\%$ threshold) |
| **H6 (B6 vs B5 Isolation)** | $\Delta w_{\text{B6}} - \Delta w_{\text{B5}}$ | $-0.1309$ | $[-0.1319, -0.1300]$ | **Supported** ($95\%$ CI strictly $< 0$) |



---

## 3. Directional Interpretation Standard

In compliance with the pre-specified statistical interpretation rule:
- Where bootstrap confidence intervals strictly exclude zero, differences are reported as statistically significant under the controlled experimental design.
- The zero-modality fail-closed and simplex conservation properties are treated as exact architectural invariants rather than empirical estimates.
