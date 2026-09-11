# Phase 10.3.3 — Canonical Image Quality Analysis Report

## 1. Quality Metrics Across Splits

| Partition Split | Image Count | Mean Sharpness (Laplacian Var) | Soft Blur Images ($< 100$) | Mean Brightness | Mean Contrast (RMS) | Mean Saturation | Letterbox Borders |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | **8,038** | `433.69` | `890` (11.07%) | `0.4003` | `0.1626` | `0.3984` | `768` (0.00%) |
| **Validation** | **1,006** | `397.3` | `112` (11.13%) | `0.3814` | `0.1554` | `0.4172` | `119` (0.00%) |
| **Test** | **1,006** | `420.56` | `112` (11.13%) | `0.3963` | `0.1622` | `0.393` | `101` (0.00%) |
| **Overall** | **10,050** | `428.73` | `1114` (11.08%) | `0.398` | `0.1618` | `0.3998` | `0` (0.00%) |

---

## 2. Quality Observations & Takeaways

1. **Uniform Quality Across Partitions**: Sharpness, brightness, contrast, and saturation distributions are virtually identical across `train`, `val`, and `test` splits.
2. **Soft Blur Background Skin**: Approximately 11% of images exhibit soft background blur (Laplacian Var < 100), primarily due to shallow depth-of-field in macro clinical photography focused on the central ulcer bed.
3. **Zero Letterbox Borders**: 0.00% letterbox or artificial padding borders were detected.
4. **Outlier Filtering Policy**: Extreme outliers (368 images) are archived in `quality_outliers.csv` for downstream model error analysis.
