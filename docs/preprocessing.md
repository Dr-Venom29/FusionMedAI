# FusionMedAI Preprocessing Documentation

## Overview

Preprocessing converts each modality's input data into the representation required by its model while keeping preprocessing specific to that modality.

Each module maintains an independent preprocessing pipeline tailored to its respective data modality.

---

## Current Module Status

| Module | Preprocessing Status |
| :--- | :--- |
| **Retina Module** | Baseline preprocessing finalized and validated through architecture benchmarking. |
| **Foot Ulcer Module** | Implemented for baseline training; further preprocessing experiments pending. |
| **Clinical Module** | Planned |
| **ACARA-U Fusion** | Planned |

---

# Retina Module

### Current Baseline

The finalized Retina Module continues to use the validated baseline preprocessing configuration established during the controlled benchmarking phase. The same preprocessing configuration was retained during architecture benchmarking, explainability, probability calibration, and uncertainty estimation to ensure experimental consistency.

The current Retina Module applies a lightweight preprocessing pipeline consisting of:

* Image resizing to 224 × 224
* Tensor conversion
* ImageNet normalization
* Standard data augmentation (Training)

This baseline intentionally avoids advanced enhancement techniques to establish a reproducible reference for future experiments.

### Planned Experiments

Future preprocessing studies include:

* Circular fundus cropping
* Black border removal
* CLAHE
* Ben Graham preprocessing
* Illumination normalization
* Dataset-specific normalization
* Higher image resolutions

---

# Foot Ulcer Module

### Current Implementation (Phase 10.2 & 10.4 Baseline)

The validated Foot Ulcer preprocessing and transform pipeline comprises:

- **Training Pipeline**:
  - Resize to 224 × 224 RGB
  - Random Rotation: $\pm 15^\circ$
  - Random Horizontal Flip: $p=0.5$
  - Color Jitter (brightness=0.2, contrast=0.2, saturation=0.1)
  - Observed dataset normalization (`mean = [0.4937, 0.3630, 0.3272]`, `std = [0.1745, 0.1632, 0.1551]`)
- **Validation / Test Pipeline**:
  - Resize to 224 × 224 RGB
  - Observed dataset normalization (100% Deterministic; 0% stochastic augmentation)

---

# Clinical Module

Planned preprocessing includes:

* Missing value handling
* Feature normalization
* Categorical encoding
* Feature engineering
* Outlier analysis
* Clinical variable standardization

Implementation pending.

---

# ACARA-U Fusion

The ACARA-U Fusion stage will operate on outputs produced by the individual modality modules.

Planned inputs include:

* Modality-level risk
* Confidence
* Reliability
* Uncertainty

Implementation pending.

---

## Design Principles

All preprocessing pipelines follow the same engineering principles:

* Modular implementation
* Reproducibility
* Configuration-driven execution
* Independent experimentation
* Verification before training

---

## Future Work

Subsequent project phases will evaluate preprocessing strategies experimentally and quantify their impact on:

* Classification performance
* Robustness
* Generalization
* Computational efficiency

The final preprocessing configuration for each module will be selected based on empirical benchmark results rather than fixed assumptions.
