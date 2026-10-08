# Statistical Bootstrap Analysis & Hypothesis Testing

## 1. Non-Parametric Bootstrap Protocol

To evaluate statistical stability without parametric distribution assumptions, a non-parametric bootstrap analysis with $B=1,000$ resamples was executed using frozen random seed $\text{seed}=115$.

For each distribution and modality combination, $95\%$ empirical confidence intervals ($[q_{0.025}, q_{0.975}]$) were estimated for:
- Mean Fused Risk $\overline{R_{\text{fusion}}}$
- Mean Dual-Constraint Risk Index $\overline{\text{DCRI}}$
- Mean Risk Shift Sensitivity $\overline{\Delta R}$

---

## 2. Empirical 95% Bootstrap Confidence Intervals (D2 Moderate Tail)

| Combination | Sample Size ($N$) | Mean Risk ($95\%$ CI) | Mean DCRI ($95\%$ CI) | Mean Sensitivity $\overline{\Delta R}$ ($95\%$ CI) |
| :--- | :---: | :---: | :---: | :---: |
| **RFC** | $175$ | $0.2952$ $[0.2715, 0.3190]$ | $0.1684$ $[0.1415, 0.1953]$ | $0.0000$ $[0.0000, 0.0000]$ |
| **RF** | $125$ | $0.3465$ $[0.3088, 0.3842]$ | $0.2268$ $[0.1865, 0.2671]$ | $0.0624$ $[0.0533, 0.0715]$ |
| **RC** | $75$ | $0.2175$ $[0.1806, 0.2544]$ | $0.2092$ $[0.1721, 0.2463]$ | $0.1084$ $[0.0855, 0.1314]$ |
| **FC** | $50$ | $0.2758$ $[0.2384, 0.3133]$ | $0.1370$ $[0.0952, 0.1788]$ | $0.1137$ $[0.0940, 0.1333]$ |
| **R** | $30$ | $0.2639$ $[0.1650, 0.3628]$ | $0.2635$ $[0.1648, 0.3625]$ | $0.1618$ $[0.1197, 0.2039]$ |
| **F** | $25$ | $0.4118$ $[0.3124, 0.5115]$ | $0.2918$ $[0.1821, 0.4015]$ | $0.2109$ $[0.1463, 0.2754]$ |
| **C** | $20$ | $0.1160$ $[0.0988, 0.1331]$ | $0.1079$ $[0.0919, 0.1240]$ | $0.2022$ $[0.1408, 0.2637]$ |

---

## 3. Pre-Specified Hypothesis Verification Outcomes

### Hypothesis H1: Simplex & Safety Invariant Preservation across Tiers
- **Formal Statement**: $\forall c \in \mathcal{C}_{\text{active}}, \forall k, \sum_{i \in \mathcal{A}_c} w_i(k) = 1.000000$ and $w_{j \notin \mathcal{A}_c}(k) = 0.000000$.
- **Observed Violation Rate**: $0 / 1,500$ evaluated decision packets across D1, D2, D3.
- **Outcome**: **CONFIRMED** (Exact mathematical invariant preserved across all tiers).

### Hypothesis H2: Tail Risk Sensitivity Across Soft Baselines
- **Formal Statement**: $D_{\text{tail}}^{\text{ACARA-U}} \le D_{\text{tail}}^{\text{B2}}, D_{\text{tail}}^{\text{B3}}, D_{\text{tail}}^{\text{B4}}, D_{\text{tail}}^{\text{B5}}$ under long-tailed distributions.
- **Observed Point Estimates**:
  - D2: ACARA-U ($0.1883$) $<$ B2 ($0.1925$), B3 ($0.1959$), B4 ($0.1967$), B5 ($0.1889$).
  - D3: ACARA-U ($0.1876$) $<$ B2 ($0.1945$), B3 ($0.1974$), B4 ($0.1973$), B5 ($0.1889$).
- **Paired Bootstrap Difference Against B5**:
  - D2 paired difference: $+0.000597$ ($95\%$ CI: $[-0.000660, 0.001839]$).
  - D3 paired difference: $+0.001239$ ($95\%$ CI: $[-0.000804, 0.003057]$).
- **Outcome**: **SUPPORTED** (ACARA-U achieved the lowest point-estimate tail sensitivity across all evaluated soft-weighting baselines in D2 and D3; the margin over B5 is small and non-significant at the $\alpha=0.05$ level).

### Hypothesis H3: Tail Dispersion Expansion via Modality Collapse
- **Formal Statement**: $\Delta \sigma(R) = \sigma(R_{\text{tail}}) - \sigma(R_{\text{head}}) > 0$.
- **Observed Values**:
  - D2: $\Delta \sigma(R) = 0.2508 - 0.1880 = +0.0628$.
  - D3: $\Delta \sigma(R) = 0.2784 - 0.1898 = +0.0885$.
- **Outcome**: **SUPPORTED DESCRIPTIVELY** — Tail-tier risk dispersion was higher than head-tier dispersion in both D2 and D3. No formal bootstrap confidence interval for the dispersion difference was pre-specified or reported, so statistical significance of this difference is not claimed.
