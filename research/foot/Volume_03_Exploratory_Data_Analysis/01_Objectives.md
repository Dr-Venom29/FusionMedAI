# 01 Objectives & Scope — Phase 10.3

## 1. Context & Scope

Phase 10.1 established the raw dataset audit and freeze, and Phase 10.2 implemented exact duplicate resolution, canonical manifest generation, source group construction, near-duplicate analysis, group-stratified splitting, lazy dataset loading, candidate image transformations, and deterministic DataLoader pipelines.

Phase 10.3 executes deep exploratory data analysis across the verified canonical modeling population of **10,050 images** (1,770 source groups) to evaluate dataset quality, class imbalance, visual signatures, class separability, non-clinical shortcuts, and outlier characteristics prior to baseline model training (Phase 10.4).

---

## 2. Technical Requirements

1. **Dataset Statistics (10.3.1)**: Measure exact pixel RGB statistics, resolution uniformity, aspect ratios, and file sizes across all 10,050 canonical images.
2. **Class Distribution (10.3.2)**: Quantify image-level and group-level class distributions across Wagner Grades 1-4 and calculate the imbalance ratio.
3. **Image Characteristics (10.3.3)**: Inspect visual characteristics, generate a $4 \times 4$ comparison grid, and document surface visual overlap risks (Grade 2 vs Grade 3).
4. **Image Quality (10.3.4)**: Measure sharpness (Laplacian variance), brightness, contrast, exposure clipping, and letterbox border presence.
5. **Outlier Analysis (10.3.5)**: Audit quality outliers across 7 dimensions and enforce a zero-deletion retention policy.
6. **Class Separability (10.3.6)**: Extract 2,048-dim ResNet50 feature embeddings, compute PCA/t-SNE 2D projections, Silhouette scores, and pairwise centroid distances.
7. **Bias & Shortcut Analysis (10.3.7)**: Audit non-clinical feature correlations (brightness, contrast, saturation, Roboflow prefixes) against Wagner grade labels.
8. **Findings & Final Decision (10.3.8)**: Establish a defensible research trail ("We observed X, therefore we decided Y because Z") and issue formal acceptance gate sign-off.
