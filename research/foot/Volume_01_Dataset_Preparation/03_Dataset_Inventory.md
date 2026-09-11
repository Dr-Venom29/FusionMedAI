# 03 — Dataset Inventory Analysis

## 1. Inventory Summary

The dataset inventory script (`src/foot/data/inventory.py`) scanned all files in `datasets/foot/raw/`.

| Metric / Property | Measured Value |
| :--- | :--- |
| **Total Scanned Files** | 10,064 |
| **Total Image Files** | 10,062 |
| **Non-Image Metadata Files** | 2 (`README.dataset.txt`, `README.roboflow.txt`) |
| **Image File Extension** | JPEG (`.jpg`) — 100.0% |
| **Color Space** | 3-Channel RGB — 100.0% |
| **Spatial Resolution** | $224 \times 224$ pixels — 100.0% uniform |
| **Total Size on Disk** | 66.71 MB |
| **Mean Image File Size** | 6.79 KB |

## 2. Partition Inventory Breakdown

| Partition Split | Grade 1 | Grade 2 | Grade 3 | Grade 4 | Total Image Files |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **train** | 2,240 | 2,345 | 2,744 | 2,310 | **9,639** |
| **valid** | 86 | 78 | 40 | 78 | **282** |
| **test** | 44 | 37 | 18 | 42 | **141** |
| **Total** | **2,370** | **2,460** | **2,802** | **2,430** | **10,062** |

## 3. Findings & Observations
1. **100% Format Uniformity**: Every image file is formatted as a $224 \times 224$ RGB JPEG image.
2. **Raw Partition Imbalance**: The raw `train` subfolder accounts for 95.79% of all files (9,639 / 10,062), while `valid` and `test` subfolders contain only 282 and 141 images respectively.
