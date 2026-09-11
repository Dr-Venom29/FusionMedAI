# 08 Dataset Bias & Shortcut Analysis — Phase 10.3.7

## 1. Feature Correlation & ANOVA Shortcut Audit

Non-clinical dimensions were evaluated against Wagner severity grades across all **10,050 canonical images**:

| Non-Clinical Dimension | Pearson Correlation ($r$) | Spearman Rank ($\rho$) | ANOVA Effect Size ($R^2$) | Risk Level |
| :--- | :---: | :---: | :---: | :---: |
| **Brightness** | **-0.1441** | -0.1612 | 0.0270 | **LOW RISK** |
| **RMS Contrast** | **+0.2406** | +0.2465 | 0.0622 | **MODERATE RISK** |
| **Saturation Mean** | **-0.0747** | -0.0689 | 0.0207 | **LOW RISK** |
| **Sharpness (Laplacian)** | **+0.1412** | +0.1385 | 0.0234 | **LOW RISK** |
| **Over-Exposure / Glare %** | **-0.0095** | -0.0121 | 0.0011 | **LOW RISK** |
| **File Size (KB)** | **+0.2446** | +0.2482 | 0.0651 | **MODERATE RISK** |
| **Letterbox Borders** | **+0.1633** | +0.1633 | 0.0357 | **LOW RISK** |

---

## 2. Roboflow Offline Augmentation Artifact Audit

- **Chi-Square & Cramer's V**: $\chi^2 = 347.87, p < 1\text{e-}70$, Cramer's $V = \mathbf{0.1316}$ (**LOW ASSOCIATION**).
- **Assessment**: Roboflow naming prefixes do not correlate strongly with class labels.
- **Leakage Prevention**: Group-stratified partitioning (Phase 10.2) placed all offline variants derived from the same source image strictly into the same partition.

---

## 3. Mitigation Directives

- Apply random color jitter ($b=0.2, c=0.2, s=0.1$) during Phase 10.4 training.
- Heatmap visualization saved to `datasets/foot/metadata/eda/figures/bias_correlation_matrix.png`.
