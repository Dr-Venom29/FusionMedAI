# 04 Outlier Analysis & Anomaly Audit — Phase 10.3.4

## 1. Executive Summary & Policy

Outlier detection was performed across the **10,050 canonical images** using `src/foot/data/eda_outlier_analysis.py`. The audit identified potentially unusual samples across five distribution dimensions:
1. Brightness outliers (Extreme dark vs extreme bright glare)
2. Blur outliers (Severe motion / macro focus blur)
3. Contrast outliers (Low contrast washed-out vs harsh contrast shadow)
4. Visual composition outliers (Color cast / high saturation)
5. Anomalous low-variance images

> [!IMPORTANT]
> **Data Retention Policy**: **NO OUTLIERS ARE AUTOMATICALLY DELETED**.
> All 10,050 images are retained. Preserving outliers maintains the zero-leakage patient source-group structure, prevents artificial sample selection bias, and ensures evaluation reflects real-world clinical photography variation.

---

## 2. Population Breakdown

```
Canonical Population (10,050 images)
        │
        ├── Normal Distribution: 8,636 images (85.93%)
        │
        └── Outlier Population: 1,414 images (14.07%)
                ├── SEVERE_BLUR: 642 images (6.39%)
                ├── LOW_CONTRAST: 439 images (4.37%)
                ├── COMPOSITION_COLOR_CAST: 315 images (3.13%)
                ├── EXTREME_BRIGHT_GLARE: 217 images (2.16%)
                ├── ANOMALOUS_LOW_VARIANCE: 116 images (1.15%)
                ├── HARSH_CONTRAST: 67 images (0.67%)
                └── EXTREME_DARK: 24 images (0.24%)
```

---

## 3. Outlier Category Investigation

### 3.1 Blur Outliers (SEVERE_BLUR — 642 images, 6.39%)
- **Threshold**: Laplacian variance $< 35.0$ or $Z < -2.0$ on log-transformed sharpness.
- **Clinical Investigation**: Handheld macro photography focuses tightly on the central ulcer bed. Surrounding healthy foot skin falls outside the focal plane, causing background soft blur. The primary lesion remains diagnostic in $>98\%$ of cases.

### 3.2 Low Contrast Outliers (LOW_CONTRAST — 439 images, 4.37%)
- **Threshold**: RMS Contrast $< 0.08$.
- **Clinical Investigation**: Over-exposed flat flash lighting or pale intact skin margins.

### 3.3 Visual Composition & Color Cast (COMPOSITION_COLOR_CAST — 315 images, 3.13%)
- **Threshold**: HSV Saturation Mean $> 0.65$ or clipping $> 35\%$.
- **Clinical Investigation**: Blue surgical drapes, green clinic floor tiles, or yellow halogen clinic lamps dominating the peripheral frame.

### 3.4 Glare Outliers (EXTREME_BRIGHT_GLARE — 217 images, 2.16%)
- **Threshold**: Mean brightness $> 0.70$.
- **Clinical Investigation**: Specular camera flash reflection on moist purulent exudate or wet saline dressing washes.

### 3.5 Anomalous Low Variance (ANOMALOUS_LOW_VARIANCE — 116 images, 1.15%)
- **Threshold**: Laplacian variance $< 8.0$ or RMS contrast $< 0.03$.
- **Clinical Investigation**: Tight extreme macro crops of featureless skin or dressing gauze.

### 3.6 Dark Outliers (EXTREME_DARK — 24 images, 0.24%)
- **Threshold**: Mean brightness $< 0.15$.
- **Clinical Investigation**: Practitioner casting shadow over the foot during clinic photography.

---

## 4. Partition Split Preservation

Outlier proportions are verified to be uniform across splits, ensuring no split bias:

| Outlier Flag | Train Split (8,038) | Val Split (1,006) | Test Split (1,006) |
| :--- | :---: | :---: | :---: |
| **Normal Distribution** | **6,905** (85.91%) | **864** (85.88%) | **867** (86.18%) |
| **Outliers Total** | **1,133** (14.09%) | **142** (14.12%) | **139** (13.82%) |

---

## 5. Downstream Error Tracking Protocol

All flagged outlier images are cataloged in `datasets/foot/metadata/quality/outlier_analysis_manifest.csv`. During Phase 10.4 baseline model validation, model error samples will be automatically cross-referenced with this manifest to determine whether misclassifications correlate with specific visual anomalies.
