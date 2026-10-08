# Head-vs-Tail Contrast & Tier Sensitivity Analysis

## 1. Mathematical Formalization of Tier Contrast Metrics

To rigorously assess performance divergences between frequent and sparse modality combinations, four contrast metrics are defined:

1. **Risk Dispersion Contrast**:
   
$$
\Delta \sigma(R) = \sigma(R_{\text{fusion}}^{\text{tail}}) - \sigma(R_{\text{fusion}}^{\text{head}})
$$

2. **DCRI Dispersion Contrast**:
   
$$
\Delta \sigma(\text{DCRI}) = \sigma(\text{DCRI}^{\text{tail}}) - \sigma(\text{DCRI}^{\text{head}})
$$

3. **Risk Sensitivity Contrast**:
   
$$
\Delta \overline{\Delta R} = \overline{\Delta R}_{\text{tail}} - \overline{\Delta R}_{\text{head}}
$$

4. **Uncertainty Contrast**:
   
$$
\Delta \overline{U_{\text{sum}}} = \overline{U_{\text{sum}}}^{\text{tail}} - \overline{U_{\text{sum}}}^{\text{head}}
$$

---

## 2. Empirical Tier Contrast Results Across Distributions

### Summary Table

| Distribution | Metric | HEAD Tier Value | MIDDLE Tier Value | TAIL Tier Value | Contrast ($\text{TAIL} - \text{HEAD}$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **D2 (Moderate)** | Mean Risk $\overline{R}$ | $0.3169 \pm 0.1880$ | $0.2397 \pm 0.1590$ | $0.2725 \pm 0.2508$ | $-0.0445$ |
| | Mean DCRI $\overline{\text{DCRI}}$ | $0.1930 \pm 0.2095$ | $0.1796 \pm 0.1669$ | $0.2305 \pm 0.2453$ | $+0.0376$ |
| | Risk Std $\sigma(R)$ | $0.1880$ | $0.1590$ | $0.2508$ | **$+0.0628$** |
| | Mean $\Delta R$ (Sensitivity) | $0.0260 \pm 0.0452$ | $0.1103 \pm 0.0891$ | $0.1883 \pm 0.1423$ | **$+0.1623$** |
| | Mean $U_{\text{sum}}$ | $0.6198 \pm 0.2487$ | $0.3002 \pm 0.3412$ | $0.2097 \pm 0.3039$ | **$-0.4101$** |
| **D3 (Strong Tail)**| Mean Risk $\overline{R}$ | $0.3175 \pm 0.1898$ | $0.2467 \pm 0.1833$ | $0.2562 \pm 0.2784$ | $-0.0613$ |
| | Mean DCRI $\overline{\text{DCRI}}$ | $0.1943 \pm 0.2113$ | $0.1893 \pm 0.1888$ | $0.2204 \pm 0.2525$ | $+0.0262$ |
| | Risk Std $\sigma(R)$ | $0.1898$ | $0.1833$ | $0.2784$ | **$+0.0885$** |
| | Mean $\Delta R$ (Sensitivity) | $0.0225 \pm 0.0448$ | $0.1045 \pm 0.0875$ | $0.1876 \pm 0.1551$ | **$+0.1651$** |
| | Mean $U_{\text{sum}}$ | $0.6162 \pm 0.2522$ | $0.2872 \pm 0.3339$ | $0.1789 \pm 0.2927$ | **$-0.4373$** |

---

## 3. Key Scientific Conclusions

1. **Observed Expansion of Risk Dispersion**: Tail combinations exhibited greater observed risk dispersion than head combinations in both D2 ($\Delta \sigma = +0.0628$) and D3 ($\Delta \sigma = +0.0885$), consistent with reduced multi-modal aggregation in lower-cardinality configurations.
2. **Observed Sensitivity Gradient**: Mean sensitivity $\overline{\Delta R}$ scaled directly with missingness tier: HEAD ($0.0225–0.0260$) $\to$ MIDDLE ($0.1045–0.1103$) $\to$ TAIL ($0.1876–0.1883$).
3. **Head-Tier Consistency Across Skew Levels**: In these controlled D2 and D3 configurations, head-tier mean risk remained highly similar ($\overline{R}_{\text{head}} = 0.3169$ under D2 vs $0.3175$ under D3) despite the change in tail frequency.
