# 06 Outlier Analysis & Anomaly Audit — Phase 10.3.5

## 1. Population Breakdown & Zero-Deletion Policy

Outlier detection was performed across the **10,050 canonical images** using `src/foot/data/eda_outlier_analysis.py`.

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

> [!IMPORTANT]
> **Data Retention Policy**: **NO OUTLIERS ARE AUTOMATICALLY DELETED**.
> Retaining 100% of outlier images preserves patient source-group integrity and ensures evaluation reflects real-world clinical photography variation.

---

## 2. Partition Proportions

| Partition | Normal Distribution | Outliers | Total | Outlier % |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 6,905 | 1,133 | 8,038 | 14.09% |
| **Val** | 864 | 142 | 1,006 | 14.12% |
| **Test** | 867 | 139 | 1,006 | 13.82% |
