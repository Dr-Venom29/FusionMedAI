# FusionMedAI
> Explainable Multi-Modal AI Framework for Diabetic Disease Analysis

Retina • Foot Ulcer • Clinical • Multimodal Fusion

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch 2.4](https://img.shields.io/badge/pytorch-2.4-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

FusionMedAI is a research framework for developing and evaluating independent AI modules for diabetic disease analysis and subsequently combining their outputs through an uncertainty-aware multimodal fusion layer.

The framework currently includes a completed retinal imaging pipeline and an independently developed diabetic foot-ulcer pipeline. The clinical module and multimodal fusion layer remain under development.

---

## Current Status

The Foot Ulcer module has completed dataset preparation, leakage-aware pipeline construction, exploratory data analysis, baseline model development, and controlled architecture benchmarking.

The architecture benchmark evaluated EfficientNet-B0, EfficientNet-B3, ConvNeXt-Tiny, Swin-Tiny, and ViT-B/16 under a common training and evaluation protocol. EfficientNet-B3 achieved the highest held-out test Macro F1 (0.6683) and was selected as the primary Foot Ulcer backbone. EfficientNet-B0 produced a nearly identical Macro F1 (0.6672) with substantially lower computational cost and is retained as a lightweight alternative.

The next stage is explainability.

### Retina Module
- ✓ Dataset preparation
- ✓ Data pipeline
- ✓ Exploratory data analysis and dataset quality assessment
- ✓ Baseline framework
- ✓ Architecture benchmarking
- ✓ Grad-CAM explainability
- ✓ Probability calibration
- ✓ Uncertainty estimation
- ✓ Module integration
- ✓ Acceptance testing

### Foot Ulcer Module
- ✓ Dataset acquisition and audit
- ✓ Canonical dataset construction
- ✓ Source-image grouping
- ✓ Duplicate and near-duplicate analysis
- ✓ Group-stratified train/validation/test splitting
- ✓ Dataset implementation
- ✓ Image preprocessing and augmentation pipeline
- ✓ DataLoader implementation
- ✓ End-to-end pipeline verification
- ✓ Statistical profiling
- ✓ Class-wise visual analysis
- ✓ Image quality analysis
- ✓ Outlier analysis
- ✓ Class separability analysis
- ✓ Dataset bias and shortcut analysis
- ✓ Baseline framework
- ✓ Architecture benchmarking
- ⬜ Explainability
- ⬜ Probability calibration
- ⬜ Uncertainty estimation
- ⬜ Module integration

### Remaining Modules
- ⬜ Clinical module
- ⬜ ACARA-U multimodal fusion

---

## Architecture

![System Architecture](docs/architecture.png)

*Figure 1. High-level architecture of the FusionMedAI framework.*

FusionMedAI is organized as a sequence of independent modality-specific pipelines followed by a multimodal fusion stage.

Each modality is developed and evaluated independently before integration. The current architecture comprises:

- **Retina Module** — diabetic retinopathy assessment from fundus images.
- **Foot Ulcer Module** — Wagner-grade classification from diabetic foot-ulcer images.
- **Clinical Module** — structured clinical risk assessment; under development.
- **ACARA-U Fusion Engine** — uncertainty- and reliability-aware aggregation of modality outputs; under development.

The fusion layer is designed to operate on modality-level risk, confidence, reliability, and uncertainty information rather than directly combining raw modality features.

---

## Research Methodology

The project follows the same general development sequence for each modality:

```text
Dataset Preparation
        ↓
Data Pipeline
        ↓
EDA & Dataset Quality
        ↓
Baseline Framework
        ↓
Architecture Benchmarking
        ↓
Explainability
        ↓
Probability Calibration
        ↓
Uncertainty Estimation
        ↓
Module Integration
        ↓
Multimodal Fusion
```

This separation is intentional. Dataset validation, model evaluation, calibration, uncertainty estimation, and integration are treated as separate research stages rather than being combined into a single training workflow.

---

## Retina Module

The Retina module has completed its full independent development cycle.

### Dataset
The module uses the APTOS 2019 diabetic retinopathy dataset. The dataset is not distributed with this repository and must be obtained separately.

### Backbone Selection
Five architectures were evaluated under a controlled benchmarking procedure:
- EfficientNet-B0
- EfficientNet-B3
- ConvNeXt-Tiny
- Swin-Tiny
- ViT-B/16

**Selected backbone**: EfficientNet-B3

EfficientNet-B3 was retained as the final Retina backbone based on the overall benchmark evaluation, including classification performance, quadratic weighted kappa, parameter count, and computational requirements.

The complete methodology and benchmark results are documented in Research Volume V — Architecture Benchmarking.

### Calibration and Uncertainty
The final Retina model uses:
- Temperature Scaling for probability calibration
- MC Dropout for predictive uncertainty estimation
- Predictive entropy
- Mutual information
- Risk-coverage analysis
- Grad-CAM for visual explanation

On the frozen Retina test set, MC predictive variance achieved an AUROC of 0.8443 for prediction-error detection. The MC Dropout configuration was evaluated for convergence, with 25 stochastic passes selected for the final implementation.

### Integration
The final Retina module combines prediction, calibrated confidence, uncertainty estimation, and Grad-CAM into a unified inference output. The integrated module has passed its acceptance tests.

---

## Foot Ulcer Module

The Foot Ulcer module has completed dataset preparation, data pipeline development, exploratory analysis, baseline evaluation, and architecture benchmarking. Explainability, calibration, uncertainty estimation, and module integration remain under development.

### Dataset

The module uses the ADPM V3.3 Diabetic Foot Ulcer Classification dataset, organized into four Wagner-based classes:

| Class | Description |
| :--- | :--- |
| **Grade 1** | Superficial ulcer |
| **Grade 2** | Deep ulcer without bone involvement |
| **Grade 3** | Deep ulcer with abscess, osteomyelitis, or joint sepsis |
| **Grade 4** | Localized gangrene |

The audited dataset contains 10,062 valid images at 224 × 224 resolution. Following duplicate filtering and source-group analysis, 10,050 canonical images were assigned to deterministic, leakage-aware splits:

- **Train**: 8,038 images
- **Validation**: 1,006 images
- **Test**: 1,006 images
- **Source Groups**: 1,770 distinct source image clusters

The final splits contain zero source-group overlap and zero exact-duplicate overlap.

### Dataset Analysis

Exploratory analysis examined statistical distributions, visual characteristics, image quality, potential shortcuts, and class separability.

The four Wagner grades are relatively balanced across the dataset. The primary visual challenge is substantial overlap between Grade 2 and Grade 3 ulcers. Quality variations and capture artifacts were retained to maintain alignment with realistic clinical imaging conditions.

### Baseline

A ResNet-50 baseline model was evaluated under the frozen source-group split using standard cross-entropy training:

- **Test Macro F1**: 0.6339
- **Balanced Accuracy**: 0.6391
- **Accuracy**: 0.6372
- **Macro ROC-AUC**: 0.8423

The baseline highlighted two key challenges: early validation performance saturation (overfitting risk) and substantial Grade 2 ↔ Grade 3 misclassification.

### Backbone Selection

Five candidate architectures were evaluated under identical, controlled experimental conditions against the ResNet-50 baseline:

| Rank | Model | Macro F1 | Balanced Accuracy | Macro ROC-AUC | Parameters | Latency (CPU) |
| :---: | :--- | ---: | ---: | ---: | ---: | ---: |
| 🥇 | **EfficientNet-B3** | **0.6683** | **0.6672** | **0.8685** | **10.70M** | **70.39 ms** |
| 🥈 | EfficientNet-B0 | 0.6672 | 0.6656 | 0.8431 | 4.01M | 37.89 ms |
| 🥉 | ConvNeXt-Tiny | 0.6566 | 0.6562 | 0.8613 | 27.82M | 109.61 ms |
| 4 | ResNet-50 (Baseline) | 0.6339 | 0.6391 | 0.8423 | 25.56M | 82.14 ms |
| 5 | Swin-Tiny | 0.6266 | 0.6262 | 0.8269 | 27.52M | 129.09 ms |
| 6 | ViT-B/16 | 0.5788 | 0.5878 | 0.8364 | 85.80M | 310.28 ms |

**Selection**: EfficientNet-B3 was selected as the primary backbone based on top performance across Macro F1, Balanced Accuracy, and Macro ROC-AUC, alongside strong Grade 2 recall (0.7114). EfficientNet-B0 is retained as a lightweight alternative (4.01M parameters, 37.89 ms latency).

### Calibration and Uncertainty

Calibration (Temperature Scaling, Vector Scaling) and uncertainty estimation (MC Dropout, Risk-Coverage Analysis) are scheduled for Phase 10.7 and Phase 10.8.

### Integration

Integrated inference combining prediction, explainability, calibration, and uncertainty estimation will be established in Phase 10.9.

---

## Core Infrastructure

The repository provides reusable infrastructure for dataset validation, model development, evaluation, and verification:

- Dataset validation
- Metadata generation
- Deterministic dataset splitting
- DataLoader and preprocessing pipelines
- Model training
- Checkpoint management
- Inference
- Architecture benchmarking
- Experiment tracking
- Grad-CAM
- Probability calibration
- MC Dropout uncertainty estimation
- Risk-coverage analysis
- Pipeline verification
- Model acceptance testing

Modality-specific implementations remain isolated under their respective source directories.

---

## Repository Structure

```directory
FusionMedAI/
├── datasets/
│   ├── retina/
│   │   ├── raw/
│   │   ├── interim/
│   │   ├── processed/
│   │   └── metadata/
│   └── foot/
│       ├── raw/
│       ├── interim/
│       ├── processed/
│       └── metadata/
├── docs/
│   └── architecture_v1.png
├── notebooks/
│   ├── retina/
│   └── foot/
├── reports/
├── research/
│   ├── retina/
│   ├── foot/
│   ├── clinical/
│   └── fusion/
├── src/
│   ├── retina/
│   └── foot/
├── verification/
│   ├── retina/
│   │   ├── data/
│   │   └── model/
│   └── foot/
│       └── data/
├── LICENSE
└── requirements.txt
```

---

## Research Documentation

### Retina
| Volume | Topic | Status |
| :--- | :--- | :---: |
| **I** | Dataset Preparation | ✅ |
| **II** | Data Pipeline | ✅ |
| **III** | Exploratory Data Analysis | ✅ |
| **IV** | Baseline Framework | ✅ |
| **V** | Architecture Benchmarking | ✅ |
| **VI** | Model Explainability | ✅ |
| **VII** | Probability Calibration | ✅ |
| **VIII** | Prediction Uncertainty Estimation | ✅ |
| **IX** | Module Integration & Finalization | ✅ |

### Foot Ulcer

| Phase | Topic | Status |
|---|---|---|
| 10.1 | Dataset Preparation & Audit | ✅ Conditional Pass |
| 10.2 | Data Pipeline | ✅ |
| 10.3.1 | Dataset Statistical Profiling | ✅ |
| 10.3.2 | Class-Wise Visual Analysis | ✅ |
| 10.3.3 | Image Quality Analysis | ✅ |
| 10.3.4 | Outlier Analysis | ✅ |
| 10.3.5 | Class Separability Analysis | ✅ |
| 10.3.6 | Dataset Bias & Shortcut Analysis | ✅ |
| 10.4 | Baseline Framework | ✅ |
| 10.5 | Architecture Benchmarking | ✅ |
| 10.6 | Explainability | ⬜ |
| 10.7 | Probability Calibration | ⬜ |
| 10.8 | Uncertainty Estimation | ⬜ |
| 10.9 | Module Integration | ⬜ |

---

## Installation & Setup

### Requirements
- Python 3.12
- PyTorch 2.4

Create a virtual environment and install project dependencies:

```bash
git clone https://github.com/Dr-Venom29/FusionMedAI.git
cd FusionMedAI

python -m venv venv

# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### Dataset Setup

#### Retina
The APTOS 2019 dataset must be obtained separately.

Expected structure:
```directory
datasets/
└── retina/
    └── raw/
        └── aptos2019/
            ├── train.csv
            └── train_images/
                ├── 000c1434d8d8.png
                ├── 001639a39701.png
                └── ...
```

#### Foot Ulcer
The Foot Ulcer dataset is maintained separately under:
```directory
datasets/
└── foot/
    └── raw/
```

The raw dataset is treated as immutable. Dataset cleaning, canonicalization, grouping, and final modeling splits are generated into the corresponding `processed/`, `interim/`, and `metadata/` directories.

---

## Verification

Verification scripts are maintained independently from the training code under `verification/`:
- `verification/retina/data/` & `verification/retina/model/`
- `verification/foot/data/`

The framework verifies components including:
- Dataset integrity
- Pipeline construction
- Model initialization
- Training and backpropagation
- Checkpoint loading
- Inference
- Explainability
- Calibration
- Uncertainty estimation
- Module-level acceptance

The project does not treat successful model training alone as sufficient validation. Each completed research stage has its own verification criteria.

---

## Example Retina Inference

### Input Fundus Scan
![Retina Input](docs/examples/retina_input.png)

### Unified Prediction & Explanation Output
![Retina Output](docs/examples/retina_output.png)

The output demonstrates the integrated Retina inference interface, including model prediction, calibrated confidence, uncertainty information, and Grad-CAM explanation.

---

## Development Roadmap

- **v1.0 (Retina Module)** — **Completed**. The Retina pipeline has progressed from dataset preparation through module integration and acceptance testing. ✅
- **v2.0 (Foot Ulcer Module)** — **In Development**. Completed: Phase 10.1 (Audit), Phase 10.2 (Pipeline), Phase 10.3 (EDA & Quality), Phase 10.4 (Baseline Framework), and Phase 10.5 (Architecture Benchmarking). Selected backbone: EfficientNet-B3. Next: **Phase 10.6 — Explainability**. ⬜
- **v3.0 (Clinical Module)** — **Planned**. Development of the independent clinical-data assessment module. ⬜
- **v4.0 (ACARA-U Fusion)** — **Planned**. Integration of the Retina, Foot Ulcer, and Clinical modules through the ACARA-U uncertainty- and reliability-aware fusion framework. ⬜

---

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE) for details.
