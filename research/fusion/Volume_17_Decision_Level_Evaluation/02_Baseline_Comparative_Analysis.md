# Chapter 02: Baseline Comparative Analysis

## 1. Comparative Benchmark Framework
To rigorously contextualize the decision behavior of ACARA-U (B6), five defined comparator policies are evaluated on the identical $N=500$ decision packets (seed 115):
1. **B1: Reliability-Selected Unimodal Baseline**: Hard winner-take-all assignment to the active channel with the highest historical reliability prior ($\beta > 0$, all other routing coefficients zero). In full tri-modal encounters where Retina, Foot, and Clinical are all active, Retina possesses prior $\beta_R = 0.85$ (vs Foot $\beta_F = 0.72$ and Clinical $\beta_C = 0.70$), so B1 assigns $w_{\text{retina}} = 1.0$ and zero weight to Foot and Clinical. In restricted regimes, B1 routes weight $1.0$ to the active channel with the highest prior among available channels.
2. **B2: Uniform Average Baseline**: Static equal allocation $w_i = \frac{1}{|\mathcal{A}|} = \frac{1}{3} \approx 0.333333$ across all active modalities in tri-modal encounters, ignoring sample-level uncertainty, quality, confidence, and priors.
3. **B3: Confidence Fusion Baseline**: Routing governed strictly by sample predictive confidence ($\alpha = 1.0, \beta = 0.0, \gamma = 0.0, \eta = 0.0$).
4. **B4: Confidence + Reliability Baseline**: Routing governed jointly by sample confidence and historical reliability priors ($\alpha = 1.0, \beta = 1.5, \gamma = 0.0, \eta = 0.0$).
5. **B5: Confidence + Reliability - Uncertainty Baseline**: Adds epistemic uncertainty down-weighting ($\alpha = 1.0, \beta = 1.5, \gamma = 1.0, \eta = 0.0$), but omits the dynamic quality bonus ($\eta = 0.0$).
6. **B6: Full ACARA-U Baseline**: The complete four-factor dynamic router ($\alpha = 1.0, \beta = 1.5, \gamma = 1.0, \eta = 0.5$).

These comparator policies are defined architectural ablations designed to isolate specific routing factors; they do not represent independently validated external clinical standards.

## 2. Quantitative Performance & Routing Attribution
Deterministic evaluation over the $N=500$ cohort yields the following observed metrics:

| Metric / Parameter | B1 (Unimodal WTA) | B2 (Uniform Avg) | B3 (Confidence) | B4 (Conf + Rel) | B5 (Conf+Rel-Unc) | B6 (Full ACARA-U) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mean Risk $\overline{R}_{\text{fusion}}$** | 0.255037 | 0.296804 | 0.314197 | 0.318442 | 0.287247 | **0.289900** |
| **Std Dev $R_{\text{fusion}}$** | 0.267237 | 0.122679 | 0.149685 | 0.152976 | 0.159944 | **0.163471** |
| **Mean DCRI $\overline{\text{DCRI}}$** | 0.191643 | 0.233410 | 0.250803 | 0.255049 | 0.223853 | **0.226506** |
| **Routing Entropy $\overline{H}$** | 0.000000 | 1.098612 | 1.061442 | 1.052013 | 1.022036 | **1.011041** |
| **Retina Weight $\overline{w}_R$** | 1.000000 | 0.333333 | 0.394419 | 0.404870 | 0.485205 | **0.501002** |
| **Foot Weight $\overline{w}_F$** | 0.000000 | 0.333333 | 0.368912 | 0.376040 | 0.259834 | **0.260925** |
| **Clinical Weight $\overline{w}_C$** | 0.000000 | 0.333333 | 0.236669 | 0.219089 | 0.254961 | **0.238073** |
| **Tier 0 Routine Review** | 275 (55.0%) | 208 (41.6%) | 202 (40.4%) | 198 (39.6%) | 234 (46.8%) | **232 (46.4%)** |
| **Tier 1 Assessment** | 84 (16.8%) | 240 (48.0%) | 208 (41.6%) | 207 (41.4%) | 186 (37.2%) | **187 (37.4%)** |
| **Tier 2 Escalation** | 141 (28.2%) | 52 (10.4%) | 90 (18.0%) | 95 (19.0%) | 80 (16.0%) | **81 (16.2%)** |
| **Negative DCRI Count** | 223 (44.6%) | 0 (0.0%) | 1 (0.2%) | 1 (0.2%) | 35 (7.0%) | **37 (7.4%)** |

## 3. Key Observations and Scope Limits

1. **Unimodal Discrepancy (B1)**: Relying exclusively on Retina produces high across-cohort standard deviation ($\sigma_R = 0.267237$) and results in 223 negative DCRI encounters, because Retina frequently predicts very low risk while the packet is penalized by the unweighted sum of all three active uncertainties ($P_U = 0.063394$).
2. **Observed Entropy Reduction**: Uniform allocation (B2) exhibits maximum theoretical entropy for tri-modal fusion ($H = \ln 3 \approx 1.098612$). Routing entropy decreases across the multi-factor policies: B3 ($1.061442$) $\to$ B4 ($1.052013$) $\to$ B5 ($1.022036$) $\to$ B6 ($1.011041$). This reduction establishes that weight allocation becomes more concentrated on subsets of channels as additional routing factors are introduced; it does not independently prove that the channels receiving greater weight perform better.
3. **Quality Bonus Shift (B6 vs B5)**: Adding $\eta = 0.5$ shifts weight slightly toward higher-quality channels ($\overline{w}_R$ increases from $0.485205$ to $0.501002$), producing a small mean risk shift from $0.287247$ to $0.289900$ and DCRI from $0.223853$ to $0.226506$.
4. **Evaluation Scope**: Higher or lower mean risk is not inherently preferable without external clinical reference outcomes and a defined screening objective. These comparisons characterize algorithmic score behavior across comparator policies; they do not establish superior clinical performance or diagnostic safety.
