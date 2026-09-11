# 06 — Image Property & Visual Statistics Audit

## 1. Measured Visual Statistics
The image property audit script (`src/foot/data/audit_properties.py`) measured channel intensity distributions across all 10,062 images.

| Color Channel | Measured Mean ($\mu$) | Measured Std ($\sigma$) | Min Intensity | Max Intensity |
| :--- | :---: | :---: | :---: | :---: |
| **Red (R)** | `0.4937` | `0.1744` | 0.0000 | 1.0000 |
| **Green (G)** | `0.3630` | `0.1632` | 0.0000 | 1.0000 |
| **Blue (B)** | `0.3272` | `0.1551` | 0.0000 | 1.0000 |
| **Overall RGB** | `0.3946` | `0.1862` | 0.0000 | 1.0000 |

## 2. Normalization Parameter Policy

```python
# Measured Foot DFU Dataset Normalization Values
OBSERVED_DATASET_MEAN = [0.4937, 0.3630, 0.3272]
OBSERVED_DATASET_STD  = [0.1744, 0.1632, 0.1551]
```

## 3. Engineering Takeaway
Normalizing images using observed Foot dataset statistics centers pixel distributions specifically for clinical wound photography, preserving skin and lesion contrast without forcing ImageNet assumptions.
