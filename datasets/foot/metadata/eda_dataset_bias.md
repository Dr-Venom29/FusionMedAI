# Phase 10.3.6 — Dataset Bias & Shortcut Analysis Report

## 1. Executive Summary

Dataset bias analysis checks for non-clinical shortcuts—such as background color, lighting, camera flash glare, framing, or Roboflow augmentation artifacts—that could correlate with Wagner severity grades (**Grade 1** through **Grade 4**) and allow models to "cheat" without learning true ulcer pathology.

Analysis was conducted across all **10,050 canonical images**.

---

## 2. Feature Correlation & ANOVA Shortcut Audit

| Non-Clinical Dimension | Pearson Correlation ($r$) | Spearman Rank ($ho$) | ANOVA $R^2$ ($\eta^2$) | Shortcut Risk Level |
| :--- | :---: | :---: | :---: | :---: |
| **Brightness** | `-0.1441` | `-0.1460` | `0.0270` | **LOW** |
| **RMS Contrast** | `+0.2406` | `+0.2513` | `0.0622` | **MODERATE** |
| **Saturation Mean** | `-0.0747` | `-0.0704` | `0.0207` | **LOW** |
| **Laplacian Sharpness** | `+0.1412` | `+0.1294` | `0.0234` | **LOW** |
| **File Size (KB)** | `+0.2446` | `+0.2357` | `0.0651` | **MODERATE** |
| **Letterbox Borders** | `+0.1633` | `+0.1634` | `0.0357` | **LOW** |

---

## 3. Class-Wise Non-Clinical Feature Profiles

| Wagner Class | Mean Brightness | Mean RMS Contrast | Mean Saturation | Mean Sharpness (Laplacian Var) | Mean File Size (KB) | Letterbox Border % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Grade 1** | `0.4299` | `0.1405` | `0.3979` | `377.4` | `6.3 KB` | `3.08%` |
| **Grade 2** | `0.393` | `0.1601` | `0.4301` | `404.27` | `6.6 KB` | `5.05%` |
| **Grade 3** | `0.3881` | `0.1694` | `0.3878` | `421.17` | `6.79 KB` | `16.2%` |
| **Grade 4** | `0.3833` | `0.1756` | `0.3847` | `512.2` | `7.46 KB` | `13.92%` |

---

## 4. Roboflow Offline Augmentation Artifact Audit

- **Cramer's V Association**: `0.1316`
- **Assessment**: Filename prefixes and Roboflow source variants show low association with Wagner grade labels ($V < 0.20$).
- **Group-Stratified Partitioning Enforcement**: Group-stratified splitting (Phase 10.2) placed all offline variants derived from the same source image strictly into the same partition, preventing offline augmentation leakage across splits.

---

## 5. Bias Mitigation Mandate for Phase 10.4 Training

1. **Color Jitter Augmentation**: Apply random brightness ($0.2$), contrast ($0.2$), and saturation ($0.1$) jitter during training to suppress residual lighting/saturation shortcuts.
2. **Observed Normalization**: Apply exact observed normalization $[0.4937, 0.3630, 0.3272]$.
