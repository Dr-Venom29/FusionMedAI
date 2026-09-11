# 06 — Group-Stratified Train/Val/Test Splitting (Phase 10.2.5)

## 1. Splitting Methodology
Script `src/foot/data/split_dataset.py` executed deterministic `StratifiedGroupKFold` partitioning using `SEED = 42`. The unit of splitting is the **source-image group** (`source_image_id`).

## 2. Partition Summary Statistics

| Partition Split | Group Count | Group Pct | Image Count | Image Pct | Class Distribution (G1 / G2 / G3 / G4) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Train** (`train.csv`) | **1,412** | **79.77%** | **8,038** | **79.98%** | **23.55% / 24.45% / 27.83% / 24.17%** |
| **Validation** (`val.csv`) | **177** | **10.00%** | **1,006** | **10.01%** | **23.56% / 24.45% / 27.83% / 24.16%** |
| **Test** (`test.csv`) | **181** | **10.23%** | **1,006** | **10.01%** | **23.56% / 24.45% / 27.83% / 24.16%** |
| **Total** | **1,770** | **100.00%** | **10,050** | **100.00%** | **Properly stratified across partitions** |

## 3. Split Files Created
Files created under `datasets/foot/processed/splits/`:
- `train.csv`: 8,038 rows
- `val.csv`: 1,006 rows
- `test.csv`: 1,006 rows
- `index.csv`: 10,050 rows
