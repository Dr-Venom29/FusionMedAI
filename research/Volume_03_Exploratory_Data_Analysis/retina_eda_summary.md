# Ocular Fundus (Retina) Dataset Exploratory Analysis Summary

## Executive Summary
This document summarizes the quantitative, colorimetric, and visual analysis of the APTOS 2019 dataset training split before model preprocessing or training.

## Dataset Fingerprint & Reproducibility
- **Dataset Fingerprint (SHA-256)**: `df102971518b36519862bbfb8b1afc601fd2ba27e6c2b2371d3c3ff77b443593`
- **EDA Module Version**: `1.0.0`
- **Execution Run Duration**: `77.27 seconds`
- **Dataset Integrity**: All 3662 images were successfully read with no missing, corrupted, or invalid-channel files.

## High-Level Statistics Table
| Metric | Value |
| :--- | :--- |
| **Total Images** | 3662 |
| **Average Width** | 2015.18 pixels |
| **Average Height** | 1526.83 pixels |
| **Resolution Range** | Min: 474x358 \| Max: 4288x2848 |
| **Average Aspect Ratio** | 1.2831 |
| **Average File Size** | 2294.22 KB |
| **Average Continuous Quality Score (Q)** | 0.7332 |
| **Detected Exact Visual Duplicate Pairs** | 134 |

## Class Distribution and Imbalance
- **Distribution Counts**:
  - Class 0 (No DR): 1805 (49.29%)
  - Class 1 (Mild): 370 (10.10%)
  - Class 2 (Moderate): 999 (27.28%)
  - Class 3 (Severe): 193 (5.27%)
  - Class 4 (Proliferative): 295 (8.06%)
- **Imbalance Ratio**: `9.3523` (Majority Class Count vs Minority Class Count)
- **Recommended Loss Function**: Class-Weighted Cross-Entropy Loss

## Global RGB Channel Mean & Standard Deviation (Normalized [0, 1])
- **Dataset Mean**: R: `0.421598` \| G: `0.223786` \| B: `0.072202`
- **Dataset Std Dev**: R: `0.276549` \| G: `0.150057` \| B: `0.080490`

The computed RGB statistics provide a dataset-specific normalization baseline that will be benchmarked against ImageNet normalization during model training.

## Quality Outlier Assessment Percentages
- **Dark Images (< 37.78)**: 5.02% (Threshold: 5th percentile)
- **Bright Images (> 92.42)**: 5.02% (Threshold: 95th percentile)
- **Blurry Images (< 24.37)**: 5.02% (Threshold: 5th percentile)
- **Corrupted Files / Invalid Channels**: Corrupted: 0 \| Missing: 0

## Candidate Preprocessing Guidelines
Based on statistical variances, the following candidate configurations have been identified for experimental evaluation during preprocessing and baseline model training:
- **Candidate Image Sizes**: `[224, 384, 512]` (To be determined via training benchmarks)
- **Candidate Normalization Methods**: `[ImageNet, Dataset]` (To be compared in ablation studies)
- **Candidate Augmentations**:
  - Horizontal Flip: `[True, False]`
  - Rotation Degrees: `[0, 15, 30]`
  - Color Jitter: `[True, False]`
  - Random Crop: `[True, False]`

## Conclusion and Next Steps
Overall, the exploratory analysis confirms that the APTOS 2019 training cohort is structurally valid, quantitatively characterized, and suitable for deep learning experimentation. The extracted metadata, quality assessments, duplicate audit, and preprocessing recommendations establish a reproducible foundation for Step 4 (Image Preprocessing) and subsequent baseline model development.
