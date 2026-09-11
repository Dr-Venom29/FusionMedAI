# Phase 10.2.8 — Candidate Image Transforms Evaluation Report

## 1. Executive Summary

This report documents the clinical and data engineering evaluation of candidate image preprocessing and data augmentation options for the Diabetic Foot Ulcer (DFU) Wagner 4-Class classification modality.

---

## 2. Measured Dataset Statistics vs Normalization Policy

| Observed Channel | Measured Dataset Mean | Measured Dataset Std | Applied Policy |
| :--- | :---: | :---: | :--- |
| **Red (R)** | `0.4937` | `0.1744` | Normalization centered on observed dataset mean & std |
| **Green (G)** | `0.3630` | `0.1632` | Normalization centered on observed dataset mean & std |
| **Blue (B)** | `0.3272` | `0.1551` | Normalization centered on observed dataset mean & std |

### Clinical Rationale
Using the observed dataset mean (`[0.4937, 0.3630, 0.3272]`) and std (`[0.1744, 0.1632, 0.1551]`) centers pixel intensity distributions appropriately for clinical DFU skin and ulcer tissue photography, avoiding forced assumption of ImageNet natural image statistics.

---

## 3. Candidate Transformation Evaluation Matrix

| Transformation Candidate | Parameter Settings | Clinical & Data Rationale | Policy Recommendation |
| :--- | :---: | :--- | :---: |
| **Bilinear Resizing** | `$224 \times 224$` | Raw images are already uniform $224 \times 224$ JPEGs. Standardizes tensor shapes. | **APPROVED** |
| **Random Rotation** | $\pm 15^\circ$ | Handheld photography in clinics exhibits natural camera tilt and patient leg positioning variance. | **APPROVED** |
| **Random Horizontal Flip** | $p = 0.5$ | **Laterality Evaluation**: Wagner grade classification evaluates ulcer tissue depth, osteomyelitis, and gangrene rather than left vs. right foot anatomical orientation. Horizontal reflection preserves diagnostic label integrity while improving spatial robustness. | **APPROVED** |
| **Color Jitter** | Brightness (0.2), Contrast (0.2), Saturation (0.1), Hue (0.02) | Simulates varying ambient clinic lighting, flash illumination, and mobile camera sensor white balances without distorting tissue color signatures (erythema, necrosis, pale tissue). | **APPROVED** |

---

## 4. Pipeline Verification

Unit tests in `src/foot/data/transforms.py` confirmed:
- Output tensor shape: `[3, 224, 224]`
- Output tensor dtype: `torch.float32`
- Deterministic behavior on validation and test splits (no random transformations applied during validation/testing).
