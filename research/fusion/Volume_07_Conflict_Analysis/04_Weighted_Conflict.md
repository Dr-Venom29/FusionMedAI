# Weighted Dispersion & Consensus Dynamics

## 1. Weighted Dispersion Formulations

The unweighted metrics ($\Delta_{\max}$ and $\Delta_{\text{mean}}$) treat all modalities identically, independent of router decision authority. In contrast, weighted variance and standard deviation incorporate the dynamic authority allocations $w_i$:

$$V_w = \sum_{i \in \mathcal{A}} w_i (r_i - R_{\text{fusion}})^2$$

$$\sigma_w = \sqrt{V_w}$$

---

## 2. Empirical Dispersion Distributions ($N=500$)

| Metric | Mean | Median | Std Dev | Min | Max | P25 | P75 | P90 | IQR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Weighted Variance ($V_w$)** | **0.058038** | 0.037022 | 0.055201 | 0.000098 | 0.230275 | 0.018418 | 0.082510 | 0.141337 | 0.064092 |
| **Weighted Std Dev ($\sigma_w$)** | **0.215019** | 0.192409 | 0.108652 | 0.009888 | 0.479869 | 0.135712 | 0.287243 | 0.375948 | 0.151532 |
| **Routing Entropy ($H$)** | **1.009745** | 1.017252 | 0.052836 | 0.864591 | 1.098058 | 0.972888 | 1.052489 | 1.077293 | 0.079601 |
| **Max Authority ($w_{\max}$)** | **0.511689** | 0.515092 | 0.074911 | 0.341835 | 0.659952 | 0.451303 | 0.573490 | 0.612823 | 0.122187 |

---

## 3. Scientific Insights on Weighted Dispersion

1. **Scale Reduction**: While maximum unweighted disagreement averages $\overline{\Delta_{\max}} = 0.5203$, weighted standard deviation averages $\overline{\sigma_w} = 0.2150$. In this controlled cohort, dynamic routing produced lower authority-weighted dispersion than the corresponding unweighted maximum disagreement, consistent with greater authority being assigned to higher-reliability and/or lower-uncertainty modalities.
2. **Entropy Moderation**: Entropy averages $1.0097$ (against the maximum uniform value $\ln 3 \approx 1.0986$), indicating sustained multimodal participation without degenerate single-modality collapse.
