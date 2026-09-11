# FusionMedAI
> Explainable Multi-Modal AI Framework for Diabetic Disease Analysis

Retina • Clinical • Foot Ulcer • Multimodal Fusion

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch 2.4](https://img.shields.io/badge/pytorch-2.4-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

FusionMedAI is a modular research framework for explainable multi-modal diabetic disease analysis. The project develops independent Retina, Clinical, and Foot Ulcer AI modules using reproducible engineering practices before integrating them through the uncertainty-aware ACARA-U Fusion Engine.

## **Current Status**

- ✓ Retina Dataset Preparation
- ✓ Retina Data Pipeline
- ✓ Retina EDA & Data Quality Analysis
- ✓ Retina Baseline Framework
- ✓ Retina Architecture Benchmarking
- ✓ Retina Explainability (Grad-CAM)
- ✓ Retina Probability Calibration
- ✓ Retina Uncertainty Estimation
- ✓ Retina Module Integration & Acceptance Testing

- ✓ Foot Ulcer Dataset Preparation & Audit
- ⬜ Foot Ulcer Data Pipeline
- ⬜ Foot Ulcer EDA & Data Quality Analysis
- ⬜ Foot Ulcer Baseline Framework
- ⬜ Foot Ulcer Architecture Benchmarking
- ⬜ Foot Ulcer Explainability
- ⬜ Foot Ulcer Probability Calibration
- ⬜ Foot Ulcer Uncertainty Estimation
- ⬜ Foot Ulcer Module Integration

- ⬜ Clinical Module
- ⬜ ACARA-U Multimodal Fusion

---

## ✨ Core Infrastructure

- ✓ Dataset verification
- ✓ Data pipeline
- ✓ EDA
- ✓ Modular training
- ✓ Benchmarking
- ✓ Experiment tracking
- ✓ Inference
- ✓ Verification
- ✓ Explainable AI (Grad-CAM)
- ✓ Probability Calibration
- ✓ Uncertainty Estimation
- ✓ Selective Prediction / Risk-Coverage Analysis
- ✓ Retina Module Integration
- ✓ Retina Module Acceptance Testing
- ✓ Foot Ulcer Dataset Audit & Decision (Phase 10.1)

---

## 🏗️ Architecture

The following diagram illustrates the multi-modal diagnostic flow of the FusionMedAI framework, from raw heterogeneous ingestion to multi-level predictive fusion:

![System Architecture](docs/architecture_v1.png)

*Figure 1: High-level architectural overview of the FusionMedAI multi-modal pipeline.*

---

## 🔬 Retina Backbone Selection

A controlled benchmark was conducted across five state-of-the-art vision architectures under identical experimental settings:

- EfficientNet-B0
- EfficientNet-B3
- ConvNeXt-Tiny
- Swin-Tiny
- ViT-B/16

**Selected Backbone: EfficientNet-B3**

EfficientNet-B3 was selected as the final Retina backbone based on its strongest overall combination of classification accuracy, QWK, parameter efficiency, and resource requirements.

ConvNeXt-Tiny achieved higher Balanced Accuracy, Macro F1, ROC-AUC, and inference throughput, but required substantially more parameters.

Complete benchmark methodology and results are available in **Research Volume 05 – Architecture Benchmarking**.

---

## 🌟 Research Highlights

- Five-model retinal architecture benchmark under identical experimental conditions
- EfficientNet-B3 selected as the final Retina backbone
- Modular PyTorch framework with reproducible experiment tracking
- Research documentation spanning nine Retina research volumes
- Grad-CAM explainability integrated into the Retina inference pipeline
- Temperature Scaling used for probability calibration
- MC Dropout uncertainty estimation validated on the frozen EfficientNet-B3 test set
- MC Predictive Variance achieved AUROC 0.8443 for prediction-error detection
- N=25 MC passes empirically supported through convergence analysis
- End-to-end Retina Module acceptance testing completed
- Foot Ulcer Phase 10.1 audit completed (10,062 images, Wagner 4-class, Conditional Pass)

---

## 🛠️ Tech Stack

**Language**
- Python 3.12

**Deep Learning**
- PyTorch
- Torchvision

**Data**
- NumPy
- Pandas

**Vision**
- OpenCV
- Albumentations

**Visualization**
- Matplotlib
- TensorBoard

---

## 🎯 Project Goals

- **Multi-Modal Diagnostic Fusion**: Integrate retinal fundus images, structured clinical features, and foot-ulcer images into a unified uncertainty-aware diagnostic framework.
- **Academic-Grade Reproducibility**: Enforce strict data validation, deterministic stratified splitting, and reproducible pipelines.
- **Fail-Fast Clinical Engineering**: Ensure dataset integrity (e.g., shape, resolution, aspect ratio, label bounds, file corruption) is programmatically verified before training begins.
- **Interpretability & Trust**: Build transparent models using Explainable AI techniques such as Grad-CAM, with advanced methods such as SHAP and Integrated Gradients reserved for future extensions.

---

## 📁 Repository Structure

```directory
FusionMedAI/
├── datasets/                 # Labeled medical databases
│   ├── retina/               # Retina dataset artifacts
│   └── foot/                 # Foot DFU dataset artifacts
├── docs/                     # Architectural diagrams & specifications
├── notebooks/                # Academic Jupyter notebooks
│   ├── retina/               # Retina notebooks & EDA
│   └── foot/                 # Foot notebooks & EDA
├── reports/                  # Benchmark, Explainability & Evaluation Reports
├── research/                 # Scientific research documentation
│   ├── retina/               # Retina Module research volumes
│   ├── foot/                 # Foot Ulcer Module research volumes
│   ├── clinical/             # Clinical Module research volumes (scalable target)
│   └── fusion/               # Multimodal fusion research volumes (scalable target)
├── src/                      # Production source codebase
│   ├── retina/               # Retina AI module
│   └── foot/                 # Foot Ulcer AI module
├── verification/             # Independent verification scripts
├── LICENSE                   # Open-source licensing
└── requirements.txt          # Virtual environment dependencies
```

---

## 📚 Research Documentation — Retina

| Volume | Topic | Status |
| :--- | :--- | :--- |
| I | Dataset Preparation | ✅ |
| II | Data Pipeline | ✅ |
| III | Exploratory Data Analysis | ✅ |
| IV | Baseline Framework | ✅ |
| V | Architecture Benchmarking | ✅ |
| VI | Model Explainability | ✅ |
| VII | Probability Calibration | ✅ |
| VIII | Prediction Uncertainty Estimation | ✅ |
| IX | Retina Module Integration & Finalization | ✅ |

Detailed documentation can be found in the `research/retina/` directory.

---

## 📚 Research Documentation — Foot Ulcer

| Phase | Topic | Status |
| :--- | :--- | :--- |
| 10.1.A | Dataset Acquisition & Structure | ✅ |
| 10.1.B | Dataset Inventory | ✅ |
| 10.1.C | Label Verification | ✅ |
| 10.1.D | Image Integrity Audit | ✅ |
| 10.1.E | Image Property Audit | ✅ |
| 10.1.F | Duplicate & Near-Duplicate Audit | ✅ |
| 10.1.G | Class Distribution & Imbalance Audit | ✅ |
| 10.1.H | Visual Quality Audit | ✅ |
| 10.1.I | Data Leakage & Patient-Case Investigation | ⚠️ Conditional |
| 10.1.J | Provenance & License Audit | ✅ |
| 10.1.K | Dataset Quality Decision | ✅ Conditional Pass |

The Foot dataset has been frozen for downstream research with mandatory source-image group-stratified splitting requirements.

Detailed documentation can be found in the `research/foot/` directory and [datasets/foot/README.md](datasets/foot/README.md).

---

## 🦶 Foot Ulcer Dataset

The Foot Ulcer module uses the ADPM V3.3 Diabetic Foot Ulcer Classification dataset, organized into four Wagner-based classes:

- **Grade 1** — Superficial Ulcer
- **Grade 2** — Deep Ulcer (without bone involvement)
- **Grade 3** — Deep Ulcer with Abscess, Osteomyelitis, or Joint Sepsis
- **Grade 4** — Localized Gangrene

The dataset contains 10,062 valid JPEG RGB images at $224 \times 224$ resolution.

The dataset passed the Phase 10.1 audit with a **Conditional Pass**.

Because the distributed dataset contains augmented variants and does not provide patient identifiers, the original train/validation/test assignments are not used as the final modeling split. Phase 10.2 will construct source-image-group-stratified splits to prevent cross-split augmentation leakage.

The immutable raw dataset is maintained under `datasets/foot/raw/`.

---

## ⚙️ Installation & Setup

### 1. Environment Setup
Verify that Python is installed (Python 3.12 recommended). Clone the repository and initialize a virtual environment:

```bash
# Clone the repository
git clone https://github.com/Dr-Venom29/FusionMedAI.git
cd FusionMedAI

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Ingestion
> Note:
> The APTOS 2019 dataset is not distributed with this repository due to Kaggle licensing. Download it separately and place it under `datasets/retina/raw/aptos2019/`.

Organize the files into the following directory layout:

```directory
datasets/
└── retina/
    └── raw/
        └── aptos2019/
            ├── train.csv
            └── train_images/
                ├── 000c1434d8d7.png
                ├── 001639a39701.png
                └── ...
```

The Foot DFU dataset is handled independently under:

```directory
datasets/
└── foot/
    └── raw/
```

The Foot raw dataset is treated as immutable and is not modified in place.

---

### 3. Retina Module Pipeline Execution

Run the Retina pipeline setup and verification scripts in order:

```bash
# Step 1: Run raw dataset checks & verify images
python src/retina/data/verify_dataset.py

# Step 2: Generate dataset metadata
python src/retina/data/generate_metadata.py

# Step 3: Compute stratified 80/10/10 train/validation/test splits
python src/retina/data/split_dataset.py

# Step 4: Execute end-to-end pipeline verification
python src/retina/data/verify_pipeline.py

# Step 5: Run Exploratory Data Analysis & report generation
python -m notebooks.retina.run_eda_analysis

# Step 6: Verify model wrapper and parameter count
python verification/model/verify_model.py

# Step 7: Verify training loop, backpropagation, and scheduler updates
python verification/model/verify_training.py

# Step 8: Verify checkpoint saving, loading, and resumes
python verification/model/verify_checkpoint.py
```

All verification and analysis steps must run successfully before proceeding to model preprocessing and training.

---

## 📊 Framework Verification

Framework verification confirmed:
- ✓ Model initialization
- ✓ Training loop
- ✓ Checkpoint recovery
- ✓ Inference
- ✓ Explainability pipeline
- ✓ Probability calibration pipeline
- ✓ Uncertainty estimation pipeline
- ✓ MC Dropout stochasticity validation
- ✓ Risk-coverage analysis
- ✓ MC convergence analysis

**Detailed results**: `reports/framework_verification.md`

---

## 🧪 Retina Module — Example Inference

The finalized Retina Module accepts a fundus image and produces a unified diagnostic output containing the predicted diabetic retinopathy class, calibrated confidence, uncertainty estimates, and a Grad-CAM explanation.

### Input Fundus Scan

![Retina Input](docs/examples/retina_input.png)

### Unified Prediction & Explanation Card

![Retina Output](docs/examples/retina_output.png)

The example above was generated by the verified Retina Module acceptance test.

---

## 🚀 Development Milestones

| Version | Status |
|---------|--------|
| v0.1.0  | Retina Dataset Preparation ✅ |
| v0.2.0  | Retina Data Pipeline ✅ |
| v0.3.0  | Retina Exploratory Data Analysis ✅ |
| v0.4.0  | Retina Baseline Model Framework ✅ |
| v0.5.0  | Retina Architecture Benchmarking ✅ |
| v0.6.0  | Retina Explainability & Probability Calibration Complete ✅ |
| v0.7.0  | Retina Uncertainty Estimation Complete ✅ |
| v1.0.0  | Retina Module Integration Complete ✅ |
| v2.0.0  | Foot Ulcer Module (Phase 10.1 Audit Complete ✅) |

---

## 🗺️ Project Roadmap

- **v0.1.0 (Retina Dataset Preparation)**: Completed raw audit, metadata generation, and resolution scanning. ✅
- **v0.2.0 (Retina Data Pipeline)**: Completed stratified split, lazy loading, transforms, and E2E verification. ✅
- **v0.3.0 (Retina Exploratory Data Analysis)**: Completed concurrent stats extraction, RGB profiling, duplicate audit, quality scoring, and automated reports. ✅
- **v0.4.0 (Retina Baseline Framework)**: Built custom model wrapper, factory, BaseClassifier, trainer, mixed precision (AMP), Early Stopping, checkpointing, standalone inference, and verification framework. ✅
- **v0.5.0 (Retina Architecture Benchmarking)**: Completed fair-benchmark comparison across 5 architectures, yielding EfficientNet-B3 as the final Retinal backbone. ✅
- **v0.6.0 (Retina Explainability & Probability Calibration)**: Completed Grad-CAM, Temperature Scaling, reliability analysis, and calibrated confidence evaluation. ✅
- **v0.7.0 (Retina Uncertainty Estimation)**: Completed MC Dropout uncertainty estimation, predictive entropy, mutual information, error-detection analysis, and risk-coverage analysis. ✅
- **v1.0.0 (Retina Module Integration)**: Completed EfficientNet-B3 integration, unified inference card, and end-to-end module acceptance verification. ✅

- **v2.0.0 (Foot Ulcer Module)**:
  Phase 10.1 Dataset Preparation & Audit completed with Conditional Pass. ✅  
  Phase 10.2 Data Pipeline is next. ⬜

- **v3.0.0 (Clinical Module)**: Build and validate the independent Clinical AI module. ⬜

- **v4.0.0 (ACARA-U Multimodal Fusion)**: Integrate Retina, Foot Ulcer, and Clinical modules through the uncertainty-aware ACARA-U fusion engine. ⬜

---

## ⚖️ License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
