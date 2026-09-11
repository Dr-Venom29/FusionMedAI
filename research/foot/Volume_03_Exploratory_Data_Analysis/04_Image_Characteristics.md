# 04 Image Characteristics & Overlap Audit — Phase 10.3.3

## 1. Overview & Visual Grid

Class-wise visual analysis was conducted using `src/foot/data/eda_class_visual_analysis.py`. A representative $4 \times 4$ visual grid contact sheet was generated and exported to `datasets/foot/metadata/eda/figures/wagner_class_comparison_grid.png`.

---

## 2. Colorimetric Profiles across Severity Grades

| Wagner Grade | Code | Canonical Samples | Mean Red ($R$) | Mean Green ($G$) | Mean Blue ($B$) | Mean Luminance ($Y$) | Std Luminance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Grade 1** | 0 | 2,367 | **0.5408** | 0.3878 | 0.3560 | **0.4299** | 0.1046 |
| **Grade 2** | 1 | 2,457 | 0.5032 | 0.3500 | 0.3246 | **0.3929** | 0.1130 |
| **Grade 3** | 2 | 2,797 | 0.4744 | 0.3576 | 0.3188 | **0.3881** | 0.1019 |
| **Grade 4** | 3 | 2,429 | **0.4602** | 0.3580 | 0.3112 | **0.3832** | 0.1148 |

- **Luminance Trend**: Mean luminance ($Y$) decreases steadily from Grade 1 ($0.4299$) to Grade 4 ($0.3832$) due to dark charcoal gangrene eschar formation.

---

## 3. Grade 2 vs Grade 3 Surface Overlap Audit

- **Grade 2**: Deep ulcer penetrating to tendon/joint without bone involvement.
- **Grade 3**: Deep ulcer accompanied by internal bone infection (osteomyelitis), deep abscess, or joint sepsis.
- **Surface Overlap Finding**: Bone infection (osteomyelitis) is an internal anatomical condition that does not always express surface exudate. On clean or dried wound beds, **Grade 2 and Grade 3 superficial ulcer beds appear visually indistinguishable**, establishing a high intrinsic surface visual overlap.
