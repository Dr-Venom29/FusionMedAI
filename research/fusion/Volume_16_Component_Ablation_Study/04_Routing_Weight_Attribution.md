# Chapter 04: Routing Weight Attribution

## 1. Attribution Methodology

Routing weight attribution decomposes how each component in the ACARA-U logit formula:
$$z_i = \alpha C_i + \beta R_i - \gamma U_i + \eta Q_i$$
shifts probability mass across the 2-simplex $\Delta^2$.

For each packet $k \in \{1, \dots, N\}$, the paired difference is computed as:
$$\Delta w_i(k) = w_i^{(Av)}(k) - w_i^{(A0)}(k)$$

Because valid variants preserve the probability simplex ($\sum_{i \in \mathcal{A}} w_i(k) = 1.0$), the sum of paired weight shifts across active channels is zero for every packet:
$$\sum_{i \in \mathcal{A}} \Delta w_i(k) \equiv 0.000000$$

Uncertainty around the mean paired difference $\overline{\Delta w_i}$ is estimated using paired non-parametric bootstrapping ($B = 1,000$ resamples, fixed seed 7116).

---

## 2. Paired Weight Shifts by Variant

The table below summarizes mean paired weight shifts ($\overline{\Delta w_i}$) and their $95\%$ bootstrap confidence intervals across the $N=500$ cohort:

| Variant | Target Component | $\overline{\Delta w_{\mathrm{retina}}}$ (95% CI) | $\overline{\Delta w_{\mathrm{foot}}}$ (95% CI) | $\overline{\Delta w_{\mathrm{clinical}}}$ (95% CI) | Mean Shift $\sum \lvert \Delta w_i \rvert$ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **A1** | No Confidence ($\alpha=0$) | $-0.070910$<br>$[-0.077364, -0.063448]$ | $-0.027963$<br>$[-0.032563, -0.023632]$ | $+0.098873$<br>$[+0.094776, +0.102611]$ | $0.197746$ |
| **A2** | No Reliability Prior ($\beta=0$) | $-0.020532$<br>$[-0.020845, -0.020164]$ | $-0.007840$<br>$[-0.008137, -0.007587]$ | $+0.028372$<br>$[+0.028033, +0.028698]$ | $0.056744$ |
| **A3** | No Uncertainty Penalty ($\gamma=0$) | $-0.083033$<br>$[-0.085288, -0.080463]$ | $+0.116634$<br>$[+0.112612, +0.120224]$ | $-0.033601$<br>$[-0.035184, -0.031941]$ | $0.233268$ |
| **A4** | No Quality Bonus ($\eta=0$) | $-0.009019$<br>$[-0.009082, -0.008944]$ | $+0.001542$<br>$[+0.001474, +0.001605]$ | $+0.007476$<br>$[+0.007367, +0.007574]$ | $0.018037$ |
| **A5** | No DCRI Discount ($\delta=0$) | $+0.000000$<br>$[+0.000000, +0.000000]$ | $+0.000000$<br>$[+0.000000, +0.000000]$ | $+0.000000$<br>$[+0.000000, +0.000000]$ | $0.000000$ |
| **A_UNIFORM** | Uniform Active Baseline | $-0.167669$<br>$[-0.175407, -0.159372]$ | $+0.072408$<br>$[+0.064782, +0.079313]$ | $+0.095260$<br>$[+0.090685, +0.099678]$ | $0.335337$ |

---

## 3. Detailed Component Attribution Analysis

### 3.1 Predictive Uncertainty Penalty ($A3, \gamma=0$)
Ablating the uncertainty penalty produces the single largest routing weight redistribution in the architecture:
- Foot ulcer weight increases dramatically by $+0.116634$ ($95\%$ CI: $[+0.112612, +0.120224]$), rising from a baseline mean of $0.252066$ to $0.368700$.
- This shift is compensated by decreases in retina ($-0.083033$) and clinical ($-0.033601$).
- **Mechanism:** In the benchmark cohort, the foot ulcer vision model exhibits the highest average predictive uncertainty ($\overline{U_{\text{foot}}} \approx 0.42$ vs $\overline{U_{\text{retina}}} \approx 0.18$ and $\overline{U_{\text{clinical}}} \approx 0.22$). Removing the $-\gamma U_i$ penalty relieves this dampening effect, allowing the foot ulcer channel to capture substantial weight.

### 3.2 Modality Confidence ($A1, \alpha=0$)
Ablating dynamic confidence produces the second largest redistribution:
- Clinical tabular weight surges by $+0.098873$ ($95\%$ CI: $[+0.094776, +0.102611]$), increasing from $0.284658$ to $0.383531$.
- Retina weight drops by $-0.070910$ ($95\%$ CI: $[-0.077364, -0.063448]$), and foot drops by $-0.027963$.
- **Mechanism:** Image model classifiers generate higher dynamic confidence spikes ($C_i$) on high-clarity encounters than tabular gradient-boosted trees. Without $\alpha C_i$, the router defaults more heavily to balanced weighting, disfavoring vision channels.

### 3.3 Reliability Prior ($A2, \beta=0$)
Ablating the historical reliability prior causes moderate reallocation:
- Clinical tabular gains $+0.028372$ ($95\%$ CI: $[+0.028033, +0.028698]$).
- Retina loses $-0.020532$, and foot loses $-0.007840$.
- **Mechanism:** The frozen reliability priors explicitly favor retina ($0.85$) and foot ($0.80$) over clinical ($0.75$). Setting $\beta=0$ removes this permanent baseline bias, moderately boosting clinical weight.

### 3.4 Input Signal Quality ($A4, \eta=0$)
Ablating the quality bonus has minimal effect:
- Retina loses $-0.009019$ ($95\%$ CI: $[-0.009082, -0.008944]$).
- Foot and clinical absorb small positive compensations ($+0.001542$ and $+0.007476$).
- **Mechanism:** Under the standard controlled benchmark cohort, signal quality is relatively homogeneous across packets ($Q_i \in [0.80, 0.95]$), yielding low variance across modalities.

### 3.5 Downstream DCRI Discount ($A5, \delta=0$)
- Invariant: $\overline{\Delta w_i} \equiv 0.000000$ for all modalities across all packets.
- **Mechanism:** The DCRI discount parameter $\delta^*$ operates downstream of router logit evaluation and weight normalization.

---

## 4. Component Attribution Hierarchy

Ordering components by total weight redistribution magnitude $\sum |\overline{\Delta w_i}|$:
1. **Uncertainty Penalty ($\gamma$):** $0.233268$
2. **Confidence Weight ($\alpha$):** $0.197746$
3. **Reliability Weight ($\beta$):** $0.056744$
4. **Quality Bonus ($\eta$):** $0.018037$
5. **DCRI Discount ($\delta$):** $0.000000$ (purely downstream)

This confirms Gate C16-08: Uncertainty and Confidence are the primary dynamic drivers of multimodal weight allocation in the ACARA-U router.
