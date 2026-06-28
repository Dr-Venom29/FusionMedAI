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
| **Detected Exact Visual Duplicate Pairs** | 0 |

## Class Distribution and Imbalance
- **Distribution Counts**:
  - Class 0 (No DR): 1805 (49.29%)
  - Class 1 (Mild): 370 (10.10%)
  - Class 2 (Moderate): 999 (27.28%)
  - Class 3 (Severe): 193 (5.27%)
  - Class 4 (Proliferative): 295 (8.06%)
- **Imbalance Ratio**: `9.3523` (Majority Class Count vs Minority Class Count)
- **Baseline Loss Function**: Cross Entropy Loss
- **Future Experiments**:
  - Weighted Cross Entropy
  - Focal Loss
  - Ordinal Loss

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
Based on statistical variances, the following configurations have been identified for experimental evaluation during preprocessing and baseline model training:
- **Candidate Image Sizes**:
  - Baseline: 224 × 224 (selected for EfficientNet-B0)
  - Future Benchmarks: 384 × 384, 512 × 512
- **Candidate Normalization Methods**:
  - Baseline: ImageNet normalization (implemented)
  - Future Evaluation: Dataset-specific normalization
- **Baseline Augmentation Strategy (Implemented)**:
  - Horizontal Flip
  - Rotation
  - Color Jitter
- **Future Experiments**:
  - Random Crop
  - Strong augmentation policies (e.g., AutoAugment, RandAugment)
  - MixUp
  - CutMix

## Baseline Decisions Adopted
Based on the exploratory analysis, the following engineering decisions were adopted during Step 4:
- Baseline input resolution: 224 × 224
- ImageNet normalization selected for baseline
- EfficientNet-B0 chosen as reference architecture
- Standard Cross Entropy Loss selected for baseline
- Dataset-specific normalization deferred for future benchmarking
- Higher input resolutions (384 and 512) reserved for comparative experiments

## Conclusion and Next Steps
The exploratory analysis successfully characterized the APTOS 2019 dataset and directly informed the baseline EfficientNet-B0 implementation. The identified preprocessing recommendations, normalization strategies, and augmentation candidates now serve as controlled experimental variables for future training, hyperparameter optimization, and architecture comparison studies.
