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

The Foot Ulcer module has completed dataset preparation, leakage-aware pipeline construction, and exploratory data analysis. The next stage is baseline model development.

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
- ⬜ Baseline framework
- ⬜ Architecture benchmarking
- ⬜ Explainability
- ⬜ Probability calibration
- ⬜ Uncertainty estimation
- ⬜ Module integration

### Remaining Modules
- ⬜ Clinical module
- ⬜ ACARA-U multimodal fusion

---

## Architecture

![System Architecture](docs/architecture_v1.png)

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

The Foot Ulcer module uses the ADPM V3.3 Diabetic Foot Ulcer Classification dataset, organized into four Wagner-based classes:

| Class | Description |
| :--- | :--- |
| **Grade 1** | Superficial ulcer |
| **Grade 2** | Deep ulcer without bone involvement |
| **Grade 3** | Deep ulcer with abscess, osteomyelitis, or joint sepsis |
| **Grade 4** | Localized gangrene |

### Dataset Construction

The audited dataset contains 10,062 valid JPEG RGB images at 224 × 224 resolution.

Following the Phase 10.1 audit, 10,050 canonical images were retained for downstream modeling. Exact duplicate handling, source-image grouping, near-duplicate analysis, and group-stratified splitting were completed during Phase 10.2.

The final modeling population consists of:

- 8,038 training images
- 1,006 validation images
- 1,006 test images
- 1,770 source-image groups

The final splits contain no source-group overlap and no exact duplicate overlap.

Patient identifiers are not provided by the distributed dataset. Therefore, patient-level separation cannot be established independently. The grouping strategy is based on recoverable source-image relationships and is documented as a dataset limitation.

The immutable raw dataset is maintained under `datasets/foot/raw/`.

### Exploratory Data Analysis

Phase 10.3 examined the canonical modeling population across statistical, visual, image-quality, outlier, class-separability, and dataset-bias dimensions.

The analysis found a relatively balanced four-class distribution and identified substantial visual overlap between Grades 2 and 3. Image-quality variation and capture-related artifacts were also documented rather than removed from the dataset.

No samples were deleted as a result of the outlier analysis.

The separability and shortcut analyses are used as diagnostic evidence for subsequent model development rather than as evidence of model performance.

Detailed analyses and generated artifacts are maintained under `research/foot/` and `datasets/foot/metadata/`.

### Current Stage
The next Foot Ulcer research stage is:

**Phase 10.4 — Foot Ulcer Baseline Framework**

Model benchmarking does not begin until the baseline framework has been established.

Detailed Foot research documentation is maintained under:
`research/foot/`

Dataset documentation is maintained under:
`datasets/foot/README.md`

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
| 10.4 | Baseline Framework | ⬜ |
| 10.5 | Architecture Benchmarking | ⬜ |
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
- **v2.0 (Foot Ulcer Module)** — **In Development**. Completed: Phase 10.1 (Audit), Phase 10.2 (Pipeline), Phase 10.3 (EDA & Quality). Next: **Phase 10.4 — Baseline Framework**. ⬜
- **v3.0 (Clinical Module)** — **Planned**. Development of the independent clinical-data assessment module. ⬜
- **v4.0 (ACARA-U Fusion)** — **Planned**. Integration of the Retina, Foot Ulcer, and Clinical modules through the ACARA-U uncertainty- and reliability-aware fusion framework. ⬜

---

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE) for details.
