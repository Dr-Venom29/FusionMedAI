# Preprocessing Documentation

## Overview

Preprocessing converts each modality's raw input data into the structured representation required by its downstream model while maintaining strict data isolation between modalities.

Each module maintains an independent preprocessing pipeline tailored to its specific data characteristics (pixel tensors for vision modalities; structured 119-dimensional feature vectors for clinical EHR data).

---

## Current Module Preprocessing Status

| Module | Preprocessing Status | Representation Contract |
| :--- | :--- | :--- |
| **Retina Module** | Finalized & Validated | $224 \times 224 \times 3$ RGB tensor, ImageNet normalized |
| **Foot Ulcer Module** | Finalized & Validated | $224 \times 224 \times 3$ RGB tensor, dataset-specific normalized |
| **Clinical Module** | Finalized & Validated (C1–C10) | Frozen 119-dimensional tabular representation contract |
| **ACARA-U Fusion** | Next Research Stage | Decision-level output schema (`ClinicalOutput`, Retina, Foot) |

---

## 1. Retina Module

### Validated Baseline Preprocessing
The finalized Retina Module applies a lightweight, reproducible preprocessing pipeline established during the controlled benchmarking phase. The same configuration was retained for explainability, probability calibration, and uncertainty estimation to avoid introducing preprocessing confounders:

- **Resize**: $224 \times 224$ pixels
- **Tensor Conversion**: Scaled to $[0, 1]$ floating-point range
- **Normalization**: ImageNet statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$)
- **Data Augmentation (Training Only)**: Random horizontal/vertical flips, subtle rotation
- **Deterministic Validation & Test**: Direct resize, tensor conversion, and normalization with zero stochastic transforms

### Future Preprocessing Experiments
Prospective preprocessing extensions for fundus imaging may investigate:
- Circular fundus cropping and black border removal
- Contrast Limited Adaptive Histogram Equalization (CLAHE)
- Ben Graham local color subtraction preprocessing
- Dataset-specific illumination normalization
- Higher spatial resolutions ($384 \times 384, 512 \times 512$)

---

## 2. Diabetic Foot Ulcer Module

### Validated Preprocessing Configuration
The validated Foot Ulcer preprocessing pipeline was utilized for baseline training, architecture benchmarking, and module integration:

#### Training Pipeline
- **Resize**: $224 \times 224$ RGB
- **Random Rotation**: $\pm 15^\circ$
- **Random Horizontal Flip**: $p = 0.5$
- **Color Jitter**: Brightness = $0.2$, Contrast = $0.2$, Saturation = $0.1$
- **Dataset-Specific Normalization**:
  - $\mu = [0.4937, 0.3630, 0.3272]$
  - $\sigma = [0.1745, 0.1632, 0.1551]$

#### Validation & Test Pipeline
- **Resize**: $224 \times 224$ RGB
- **Dataset Normalization**: Applied identically using training statistics
- **Stochastic Augmentation**: None (100% deterministic evaluation)

---

## 3. Structured Clinical Tabular Module

### Validated Clinical Preprocessing (Phase C4 & C10)
The Clinical Module operates on the frozen **119-dimensional representation** established by the clinical modeling pipeline (`ClinicalPreprocessor` with standard scaling on numerical columns):

- **Schema & Feature-Order Validation**: Strict enforcement of required columns and ordering.
- **Missing-Value Imputation**: Deterministic zero-imputation / grouped categorical assignment according to the frozen preprocessor state.
- **Physiological Bounds Validation**: Range checks for encounter lengths, lab counts, procedure counts, and medication exposures.
- **Representation Dimensionality**: Exact verification of $D=119$ features.
- **Error Handling**: Input validation layer rejects malformed encounters via `ClinicalValidationError` with zero unhandled runtime exceptions.

The clinical preprocessing and inference contract is frozen and verified across single-encounter and batch workflows ($N=14,913$).

---

## 4. ACARA-U Multimodal Fusion

ACARA-U is the next research stage after independent validation of the Retina, Foot Ulcer, and Clinical modules.

The fusion layer will operate on **modality-level outputs** rather than raw patient features:
- Modality-specific point predictions and class logits
- Validation-calibrated probabilities ($p_{\text{cal}}$)
- Predictive uncertainty dispersion ($\sigma_p$, predictive entropy)
- Feature and spatial attributions (TreeSHAP rankings, Grad-CAM maps)
- Data-quality and distribution-shift alerts

*Note: The exact ACARA-U input contract will be defined and verified during the fusion experiments. No multimodal fusion results are currently reported.*

---

## Design Principles

All preprocessing pipelines follow strict engineering and scientific principles:
- **Modular Implementation**: Preprocessing logic is encapsulated within modality-specific packages (`src/retina/`, `src/foot/`, `src/clinical/`).
- **Reproducibility**: Deterministic evaluation on validation and test partitions.
- **Configuration-Driven Execution**: Preprocessing parameters are declared in versioned configuration files.
- **Verification at Each Major Stage**: Input validators and transform pipelines are verified by automated test suites.

---

## Future Work

Future preprocessing studies will evaluate alternative transformations within the individual modality pipelines where scientifically justified:
- Impact of artifact filtering on calibration quality.
- Influence of normalization techniques on predictive uncertainty.
- Computational efficiency and inference latency profiling.

Changes to preprocessing will be evaluated comparatively against the established frozen baselines.
