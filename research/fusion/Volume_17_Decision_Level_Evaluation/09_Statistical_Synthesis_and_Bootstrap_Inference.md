# Chapter 09: Statistical Synthesis and Bootstrap Inference

## 1. Statistical Methodology & Paired Bootstrap Framework
Phase C11.17 employs non-parametric paired bootstrap resampling to quantify estimation uncertainty and evaluate whether observed score and weight differences between reference ACARA-U (B6) and comparator policies (B1–B5) are distinguishable from zero:
- **Resampling Iterations**: $B = 1,000$ paired bootstrap resamples.
- **Random Number Generator**: `numpy.random.default_rng(7117)` for exact reproducibility.
- **Nominal Significance Level**: $\alpha = 0.05$ (95% confidence intervals).
- **Quantile Convention**: Empirical percentile interval $\text{CI}_{0.95} = \left[ \text{Percentile}_{2.5}(\hat{\theta}^*), \text{Percentile}_{97.5}(\hat{\theta}^*) \right]$.
- **Significance Criterion**: A difference is declared statistically distinguishable from zero at the nominal $\alpha = 0.05$ level if the 95% bootstrap confidence interval strictly excludes zero ($0 \notin [\text{CI}_{\text{lower}}, \text{CI}_{\text{upper}}]$). The pipeline does not compute exact analytical p-values; statements of statistical significance indicate non-overlap of the 95% percentile interval with zero.
- **Paired Resampling Unit**: Resampling is performed by indexing shared packet identifiers (`packet_id`) with replacement. For each bootstrap iteration, the identical sample of $N=500$ packets is evaluated across both policies, and the paired difference $\Delta(k) = \theta_{\text{B6}}(k) - \theta_{\text{comparator}}(k)$ is calculated within each sampled encounter.

## 2. Paired Differences: ACARA-U (B6) vs Baseline Comparators
All point estimates, standard errors, and confidence intervals below are extracted directly from the canonical statistical artifact (`06_statistical_analysis.json`):

### 2.1 Fused Risk Differences ($\Delta_{R_{\mathrm{fusion}}}$)
| Comparator Baseline | Point Estimate $\Delta$ | 95% Bootstrap CI | Standard Error (SE) | Excludes Zero? | Statistically Significant ($\alpha=0.05$)? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **B6 vs B1 (Unimodal WTA)** | +0.034863 | [+0.020646, +0.050332] | 0.007639 | YES | **YES** ($0 \notin \text{CI}$) |
| **B6 vs B2 (Uniform Avg)** | -0.006904 | [-0.012776, -0.001718] | 0.002780 | YES | **YES** ($0 \notin \text{CI}$) |
| **B6 vs B3 (Confidence)** | -0.024297 | [-0.027962, -0.021096] | 0.001762 | YES | **YES** ($0 \notin \text{CI}$) |
| **B6 vs B4 (Conf + Rel)** | -0.028543 | [-0.031975, -0.025413] | 0.001699 | YES | **YES** ($0 \notin \text{CI}$) |
| **B6 vs B5 (Conf+Rel-Unc)** | +0.002653 | [+0.002255, +0.003063] | 0.000208 | YES | **YES** ($0 \notin \text{CI}$) |

### 2.2 Dual-Constraint Risk Index Differences ($\Delta_{\mathrm{DCRI}}$)
Because the unweighted uncertainty penalty $P_U = \delta^* \sum_{i \in \mathcal{A}} U_i$ is calculated from identical channel inputs and discount parameter ($\delta^* = 0.10$) across all policies on any given packet, the penalty cancels out in paired differences:
$$\Delta_{\text{DCRI}} = (R_{\text{fusion, B6}} - P_U) - (R_{\text{fusion, comp}} - P_U) \equiv \Delta_{R_{\text{fusion}}}$$
- **B6 vs B5 $\Delta_{\mathrm{DCRI}}$**: $+0.002653$ (95% CI: $[+0.002255, +0.003063]$, SE: $0.000208$, excludes zero).

### 2.3 Modality Weight Allocation Shifts (B6 vs B5 Quality Bonus Attribution)
Because B6 and B5 share identical inputs, historical reliability priors ($\beta = 1.5$), dynamic confidence weights ($\alpha = 1.0$), and uncertainty penalties ($\gamma = 1.0$), the B6–B5 comparison strictly isolates the contribution of the adaptive quality bonus weight ($\eta = 0.5$ in B6 vs $\eta = 0.0$ in B5):
- **Retina Weight Shift $\Delta w_R$**: $+0.015797$ (95% CI: $[+0.015611, +0.015961]$, SE: $0.000089$).
- **Foot Weight Shift $\Delta w_F$**: $+0.001091$ (95% CI: $[+0.000960, +0.001237]$, SE: $0.000070$).
- **Clinical Weight Shift $\Delta w_C$**: $-0.016888$ (95% CI: $[-0.017084, -0.016677]$, SE: $0.000106$).
- **Routing Entropy Shift $\Delta H$**: $-0.010994$ (95% CI: $[-0.011426, -0.010533]$, SE: $0.000227$).

## 3. Scope of Statistical Conclusions
1. All evaluated pairwise differences in fused risk and DCRI are statistically distinguishable from zero at the nominal 95% confidence level under the declared paired bootstrap procedure.
2. The dynamic quality bonus ($\eta = 0.5$) systematically reallocates a small fraction of routing weight toward higher-quality imaging modalities (Retina $+1.58\%$, Foot $+0.11\%$) from structured clinical tabular inputs ($-1.69\%$), producing a small, consistent upward shift in mean fused risk ($+0.002653$).
3. These differences establish that routing factor modifications produce measurable, statistically stable changes in weight allocation and risk scoring. Without independent clinical reference outcomes, these score shifts cannot be interpreted as demonstrations of superior clinical diagnostic performance or enhanced patient safety.
