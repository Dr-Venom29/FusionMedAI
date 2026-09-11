# 02 Dataset Statistics — Phase 10.3.1

## 1. Overview

Dataset statistics were calculated across all **10,050 canonical images** using `src/foot/data/eda_statistical_profiling.py`. Artifacts are stored at `datasets/foot/metadata/eda/statistics/`.

---

## 2. Quantitative Summary

### 2.1 Resolution & Dimensions

- **Total Images**: 10,050 / 10,050 (100.00% success)
- **Width**: 224 px (100.00% uniform)
- **Height**: 224 px (100.00% uniform)
- **Aspect Ratio**: 1.0000 (100.00% square)
- **Channels**: 3 (RGB, 100.00% uniform)

### 2.2 Color Statistics (RGB Channel Mean & Std)

| Channel | Mean | Standard Deviation | Min Channel Mean | Max Channel Mean |
| :--- | :---: | :---: | :---: | :---: |
| **Red (R)** | **0.4937** | **0.1745** | 0.0821 | 0.9412 |
| **Green (G)** | **0.3630** | **0.1632** | 0.0512 | 0.8923 |
| **Blue (B)** | **0.3272** | **0.1551** | 0.0389 | 0.8712 |

### 2.3 Disk File Sizes

- **Mean File Size**: **6.79 KB** per image
- **Median File Size**: 6.64 KB
- **Total Dataset Size**: 68.24 MB

---

## 3. Partition Split Preservation

| Split | Image Count | Mean R | Mean G | Mean B | Std R | Std G | Std B | Mean File Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 8,038 | 0.4941 | 0.3632 | 0.3274 | 0.1746 | 0.1633 | 0.1552 | 6.80 KB |
| **Val** | 1,006 | 0.4912 | 0.3614 | 0.3255 | 0.1738 | 0.1624 | 0.1542 | 6.74 KB |
| **Test** | 1,006 | 0.4932 | 0.3628 | 0.3271 | 0.1742 | 0.1630 | 0.1549 | 6.78 KB |
