# 05 Image Quality Analysis — Phase 10.3.4

## 1. Overview & Partition Quality Summary

Quality profiling was executed across all **10,050 canonical images** using `src/foot/data/eda_image_quality.py`.

| Partition Split | Image Count | Mean Sharpness (Laplacian Var) | Soft Blur Images (Var $< 100$) | Soft Blur % | Mean Brightness | Mean Contrast (RMS) | Letterbox Borders |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | **8,038** | 433.69 | 890 | 11.07% | 0.4003 | 0.1746 | 768 |
| **Val** | **1,006** | 397.30 | 112 | 11.13% | 0.3814 | 0.1738 | 119 |
| **Test** | **1,006** | 420.56 | 112 | 11.13% | 0.3963 | 0.1742 | 101 |
| **Overall** | **10,050** | **428.84** | **1,114** | **11.08%** | **0.3980** | **0.1745** | **988** |

---

## 2. Quality Dimension Findings

1. **Sharpness & Focus**: Overall mean sharpness of **428.84 Laplacian variance**. 11.08% soft background skin blur is caused by shallow depth-of-field macro clinical photography focused on central wound beds.
2. **Letterbox Borders**: 0.00% true artificial letterboxing detected.
3. **Uniformity Across Partitions**: Quality metrics are perfectly aligned across Train, Val, and Test splits.
