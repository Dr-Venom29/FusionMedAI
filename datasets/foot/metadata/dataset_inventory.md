# Dataset Inventory Report: Foot DFU (Wagner 4-Class)

## Summary Statistics
```text
Dataset Inventory
────────────────────────────────────────────
Total files:       10,064
Total images:      10,062
Total size:        66.71 MB
Mean file size:    6.79 KB

Classes (Overall):
  Grade 1         : 2,370
  Grade 2         : 2,460
  Grade 3         : 2,802
  Grade 4         : 2,430

Formats:
  JPEG            : 10,062

Color Channels:
  3-channel (RGB) : 10,062

Top Image Dimensions:
  224 × 224       : 10,062
```

## Breakdown by Split

| Split | Grade 1 | Grade 2 | Grade 3 | Grade 4 | Total Images |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **test** | 44 | 37 | 18 | 42 | **141** |
| **train** | 2,240 | 2,345 | 2,744 | 2,310 | **9,639** |
| **valid** | 86 | 78 | 40 | 78 | **282** |

## Folder Structure
```directory
datasets/foot/raw/
├── test
├── test/Grade 1
├── test/Grade 2
├── test/Grade 3
├── test/Grade 4
├── train
├── train/Grade 1
├── train/Grade 2
├── train/Grade 3
├── train/Grade 4
├── valid
├── valid/Grade 1
├── valid/Grade 2
├── valid/Grade 3
├── valid/Grade 4
```

## File Extension Breakdown

| Extension | Count |
| :--- | :--- |
| `.jpg` | 10,062 |
| `.txt` | 2 |