# FusionMedAI Preprocessing Documentation

## Overview

Preprocessing is a critical stage of the FusionMedAI pipeline. Its objective is to convert heterogeneous clinical data into standardized representations suitable for deep learning while preserving clinically relevant information.

Each module maintains an independent preprocessing pipeline tailored to its respective data modality.

---

## Current Module Status

| Module            | Preprocessing Status               |
| ----------------- | ---------------------------------- |
| Retina Module     | Baseline preprocessing implemented |
| Foot Ulcer Module | Planned                            |
| Clinical Module   | Planned                            |
| ACARA-U Fusion    | Planned                            |

---

# Retina Module

### Current Baseline

The current Retina Module applies a lightweight preprocessing pipeline consisting of:

* Image resizing
* Tensor conversion
* ImageNet normalization
* Standard data augmentation

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

Planned preprocessing includes:

* Color normalization
* Hair and artifact removal (if required)
* Contrast enhancement
* Lesion cropping
* Image resizing
* Data augmentation

Implementation pending.

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

Before multimodal fusion, features extracted from individual modules will undergo:

* Feature normalization
* Dimensionality alignment
* Embedding projection
* Cross-modal attention preparation
* Uncertainty calibration

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
