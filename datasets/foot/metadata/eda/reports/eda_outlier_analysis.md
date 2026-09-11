# Phase 10.3.4 — Outlier Analysis & Anomaly Investigation

## 1. Overview & Policy

Outlier detection was performed across all **10,050 canonical images** to identify samples lying outside normal distributional bounds in brightness, sharpness, contrast, exposure, or color composition.

> [!IMPORTANT]
> **Data Retention Policy**: **NO OUTLIERS ARE AUTOMATICALLY DELETED**.
> All 10,050 images remain in the dataset. Retaining outliers preserves the zero-leakage patient source-group structure, avoids artificial data truncation, and ensures evaluation accurately reflects real-world clinical photography variations (e.g., flash glare, shadow, hand-held camera blur).

---

## 2. Outlier Distribution Summary

- **Normal Distribution Population**: **8,636 images** (85.93%)
- **Total Outlier Population**: **1,414 images** (14.07%)

### Breakdown by Outlier Flag

| Outlier Flag | Description | Sample Count | % of Population | Primary Clinical Cause |
| :--- | :--- | :---: | :---: | :--- |
| **SEVERE_BLUR** | Laplacian Variance $< 35.0$ | **642** | 6.39% | Soft macro focus on central wound; soft background skin. |
| **EXTREME_DARK** | Brightness $< 0.15$ | **24** | 0.24% | Dark clinical room illumination / shadow cast by practitioner. |
| **LOW_CONTRAST** | RMS Contrast $< 0.08$ | **439** | 4.37% | Over-exposed flat flash lighting or pale skin background. |
| **COMPOSITION_COLOR_CAST** | Saturation $> 0.65$ or clipping | **315** | 3.13% | Intense yellow/blue clinical lighting or surgical drapes. |
| **EXTREME_BRIGHT_GLARE** | Brightness $> 0.70$ | **217** | 2.16% | Direct flash reflection on moist wound bed or white sheets. |
| **HARSH_CONTRAST** | RMS Contrast $> 0.32$ | **67** | 0.67% | Direct spotlight with deep shadows. |
| **ANOMALOUS_LOW_VARIANCE** | Laplacian $< 8.0$ or Contrast $< 0.03$ | **116** | 1.15% | Extremely cropped or uniform texture images. |

---

## 3. Split Distribution Uniformity

Outliers are proportionally distributed across partitions, confirming no split bias:

| Outlier Flag | Train Outliers | Val Outliers | Test Outliers |
| :--- | :---: | :---: | :---: |
| **SEVERE_BLUR** | 512 | 67 | 63 |
| **EXTREME_DARK** | 23 | 0 | 1 |
| **LOW_CONTRAST** | 344 | 62 | 33 |

---

## 4. Modeling Directives for Outliers

1. **Robust Preprocessing**: Standard PyTorch ImageNet / Dataset Normalization ($[0.4937, 0.3630, 0.3272]$) effectively handles brightness/contrast variations.
2. **Error Tracking**: Outlier image paths in `outlier_analysis_manifest.csv` will be cross-referenced during Phase 10.4+ validation error analysis to determine if model failures correlate with visual anomalies.
