# Comparative Baseline Ladder Tail Analysis

## 1. Baseline Architectures Evaluated

Under Phase C11.9, the 6 standard decision fusion architectures are evaluated across the identical packet allocations for distributions D1, D2, and D3:

- **B1 (Reliability-Selected / Winner-Take-All)**: Assigns weight $1.0$ to the single active channel with highest reliability prior $R_i$.
- **B2 (Uniform Average)**: $w_i = 1/|\mathcal{A}|$ across active channels.
- **B3 (Confidence-Weighted)**: Softmax over active channel confidence $C_i$.
- **B4 (Confidence + Reliability)**: Softmax over $\alpha C_i + \beta R_i$.
- **B5 (Conf + Rel - Uncertainty)**: Softmax over $\alpha C_i + \beta R_i - \gamma U_i$.
- **B6 (ACARA-U Full Router)**: Softmax over $\alpha C_i + \beta R_i - \gamma U_i + \eta Q_i$.

---

## 2. Comparative Tail Sensitivity ($D_{\text{tail}}$)

![Figure 9.3: Comparative Tail-Tier Sensitivity Across Fusion Baselines](figures/fig9_3_baseline_tail_comparison.png)

The tail sensitivity metric measures the mean absolute deviation of tail-tier predictions from their reference tri-modal full fusion:

$$
D_{\text{tail}} = \frac{1}{N_{\text{tail}}} \sum_{k \in \text{TAIL}} |R_{\text{fusion}}^{\text{model}}(k) - R_{\text{fusion}}^{\text{RFC, model}}(k)|
$$

### Comparative Performance Table Under Moderate (D2) and Strong (D3) Long-Tail Regimes

| Baseline ID | Architecture Name | D2 Head $\overline{R}$ ($\sigma$) | D2 Tail $\overline{R}$ ($\sigma$) | D2 Tail Sensitivity $D_{\text{tail}}$ | D3 Head $\overline{R}$ ($\sigma$) | D3 Tail $\overline{R}$ ($\sigma$) | D3 Tail Sensitivity $D_{\text{tail}}$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1** | Reliability-Selected | $0.2547$ ($0.2755$) | $0.2725$ ($0.2508$) | $0.1786$ | $0.2606$ ($0.2726$) | $0.2562$ ($0.2784$) | $0.1577$ |
| **B2** | Uniform Average | $0.3435$ ($0.1532$) | $0.2725$ ($0.2508$) | $0.1925$ | $0.3371$ ($0.1539$) | $0.2562$ ($0.2784$) | $0.1945$ |
| **B3** | Confidence-Weighted | $0.3468$ ($0.1692$) | $0.2725$ ($0.2508$) | $0.1959$ | $0.3459$ ($0.1738$) | $0.2562$ ($0.2784$) | $0.1974$ |
| **B4** | Confidence + Reliability | $0.3490$ ($0.1706$) | $0.2725$ ($0.2508$) | $0.1967$ | $0.3487$ ($0.1753$) | $0.2562$ ($0.2784$) | $0.1973$ |
| **B5** | Conf + Rel - Uncertainty | $0.3175$ ($0.1860$) | $0.2725$ ($0.2508$) | $0.1889$ | $0.3177$ ($0.1880$) | $0.2562$ ($0.2784$) | $0.1889$ |
| **B6** | **ACARA-U (Full Router)** | **0.3169** ($0.1880$) | **0.2725** ($0.2508$) | **0.1883** | **0.3175** ($0.1898$) | **0.2562** ($0.2784$) | **0.1876** |

---

## 3. Methodological Interpretations & Insights

1. **Observed Point-Estimate Sensitivity Across Evaluated Soft-Weighting Baselines**:
   - Under D2 (Moderate Tail): ACARA-U ($D_{\text{tail}} = 0.1883$) produced lower observed point-estimate tail sensitivity than Uniform Averaging B2 ($0.1925$), Confidence-Weighted B3 ($0.1959$), Confidence+Reliability B4 ($0.1967$), and Conf+Rel-Uncertainty B5 ($0.1889$).
   - Under D3 (Strong Tail): ACARA-U ($D_{\text{tail}} = 0.1876$) produced lower observed point-estimate tail sensitivity than B2 ($0.1945$), B3 ($0.1974$), B4 ($0.1973$), and B5 ($0.1889$).
2. **Paired Bootstrap Difference Analysis Against B5**:
   - While ACARA-U achieved the lowest point estimate among soft-allocation strategies, the margin over B5 was small:
     - D2: $\overline{\Delta D} = D_{\text{tail}}^{\text{B5}} - D_{\text{tail}}^{\text{ACARA-U}} = +0.000597$ ($95\%$ paired bootstrap CI: $[-0.000660, 0.001839]$).
     - D3: $\overline{\Delta D} = D_{\text{tail}}^{\text{B5}} - D_{\text{tail}}^{\text{ACARA-U}} = +0.001239$ ($95\%$ paired bootstrap CI: $[-0.000804, 0.003057]$).
   - Because the paired bootstrap confidence intervals cross zero, the difference between B6 and B5 should be described as a consistent directional point estimate rather than a statistically significant separation.
3. **Behavior of Winner-Take-All Baseline B1**:
   - Baseline B1 achieves a lower nominal $D_{\text{tail}}$ ($0.1786$ under D2, $0.1577$ under D3) because it selects Retina exclusively whenever Retina is active, ignoring multi-channel blending. Consequently, its head dispersion is drastically elevated ($\sigma = 0.2755$ in D2 vs $0.1880$ for ACARA-U; $\sigma = 0.2726$ in D3 vs $0.1898$ for ACARA-U), eliminating multi-modal synergy in complete observations. B1 serves as a non-soft structural reference baseline.
4. **Synergistic Stabilization via Quality Layer**:
   - The inclusion of the image/tabular quality term $\eta Q_i$ in ACARA-U (B6 vs B5) provides a modest directional stabilization in tail perturbation without compromising head dispersion.
