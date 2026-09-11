# Phase 10.1.G — Class Distribution & Imbalance Audit Report

## Executive Summary
- **Total Images Measured**: `10,062`
- **Majority Class**: `Grade 3` (2,802 images, 27.85%)
- **Minority Class**: `Grade 1` (2,370 images, 23.55%)
- **Overall Imbalance Ratio**: `1.18 : 1`
- **Assessment Status**: `WELL_BALANCED`

## Overall Class Distribution
| Wagner Grade | Class Index | Image Count | Percentage | Distribution Visual |
| :--- | :---: | :---: | :---: | :--- |
| **Grade 1** | `0` | 2,370 | 23.55% | `████████████` |
| **Grade 2** | `1` | 2,460 | 24.45% | `████████████` |
| **Grade 3** | `2` | 2,802 | 27.85% | `██████████████` |
| **Grade 4** | `3` | 2,430 | 24.15% | `████████████` |

## Partition Split Level Breakdown
| Partition Split | Total Images | Grade 1 | Grade 2 | Grade 3 | Grade 4 | Split Imbalance Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **test** | 141 | 44 (31.21%) | 37 (26.24%) | 18 (12.77%) | 42 (29.79%) | `2.44 : 1` |
| **train** | 9,639 | 2,240 (23.24%) | 2,345 (24.33%) | 2,744 (28.47%) | 2,310 (23.97%) | `1.23 : 1` |
| **valid** | 282 | 86 (30.5%) | 78 (27.66%) | 40 (14.18%) | 78 (27.66%) | `2.15 : 1` |

## Key Observations & Scientific Rationale
1. **Overall Raw Balance**: The overall raw dataset is remarkably well-balanced (Imbalance Ratio = `1.18 : 1`), with all 4 classes representing between 23.5% and 27.8% of the total dataset.
2. **Existing Split Skew**: The raw `valid` and `test` folders show higher class variance (`valid` imbalance ratio = `2.15 : 1`, `test` imbalance ratio = `2.44 : 1`), where `Grade 3` is underrepresented in valid/test relative to `train`.
3. **Downstream Strategy**: In Step 10.2 (Data Pipeline), we will construct stratified splits to ensure uniform class proportions across train/validation/test partitions.

> **Measurement Notice**: Class balance was measured strictly as observed in the raw directory without altering sampling or loss functions.