# Phase 10.1.E — Image Property Audit & Preprocessing Rationale

## Executive Summary
- **Total Images Analyzed**: `10,062`
- **Uniform Resolution**: `224 × 224` (100% Aspect Ratio = 1.0)
- **Color Space**: 100% `3-Channel RGB`
- **File Format**: 100% `JPEG`

## Image Dimensions & Aspect Ratio
| Property | Value |
| :--- | :--- |
| **Width Range** | 224 px – 224 px |
| **Height Range** | 224 px – 224 px |
| **Aspect Ratio** | 1.0 (Square `224 × 224`) |

## Computed Visual Statistics
| Metric | Red Channel | Green Channel | Blue Channel | Overall |
| :--- | :---: | :---: | :---: | :---: |
| **Dataset Mean** | 0.4937 | 0.363 | 0.3272 | 0.3946 |
| **Dataset Std** | 0.1744 | 0.1632 | 0.1551 | 0.1862 |

## File Properties & Size Distribution
| File Property | Statistic |
| :--- | :--- |
| **Min File Size** | 2.4 KB |
| **Max File Size** | 17.26 KB |
| **Mean File Size** | 6.79 KB |
| **Median File Size** | 6.6 KB |

## Foot DFU Preprocessing Rationale & Candidate Options
> **Crucial Finding**: Do NOT copy Retina preprocessing assumptions (e.g. heavy cropping / Ben Graham green-channel filtering). The Foot DFU dataset consists of pre-resized $224 \times 224$ RGB clinical wound photos with natural lighting and skin tones.

### 1. Observed Dataset Statistics
```python
OBSERVED_DATASET_MEAN = [0.4937, 0.363, 0.3272]
OBSERVED_DATASET_STD  = [0.1744, 0.1632, 0.1551]
```

### 2. Candidate Preprocessing Options (To be evaluated in Step 10.2 - Data Pipeline)
- **Target Resolution**: Native `$224 \times 224$` resolution (no aspect ratio distortion).
- **Candidate Rotation Range**: `[-15°, +15°]`
- **Candidate Horizontal Flip**: `p = 0.5`
- **Candidate Color Jitter**: Brightness (0.2), Contrast (0.2), Saturation (0.1).