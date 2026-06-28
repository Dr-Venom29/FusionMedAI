# FusionMedAI: Multi-Modal Clinical Intelligence System

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch 2.4](https://img.shields.io/badge/pytorch-2.4-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

FusionMedAI is a clinical engineering framework designed for multi-modal medical diagnostics and interpretability. The system targets diagnostic classification across heterogeneous datasets (imaging, clinical structured data, and patient reports) by combining advanced deep learning backbones with transparent explainable AI (XAI) models.

The primary architecture utilizes **ACARA-U Fusion** (Attention-driven Clinical and Retinal Analysis with Uncertainty estimation) to fuse high-resolution clinical imaging, tabular EHR markers, and visual explanation metrics into robust diagnostic predictions.

## ✨ Current Features

- Automated dataset verification and metadata generation
- Stratified train/validation/test splitting (80/10/10)
- Custom PyTorch Dataset and configurable DataLoader
- End-to-end pipeline verification
- Exploratory Data Analysis (EDA)
- RGB channel profiling and quality assessment
- Duplicate image auditing (SHA-256)
- EfficientNet-B0 baseline implementation
- Modular PyTorch training framework
- Automatic checkpoint management
- TensorBoard experiment tracking
- Versioned experiment management
- Comprehensive evaluation metrics (QWK, Macro F1, Balanced Accuracy)
- Standalone inference API (single & batch prediction)
- Automated framework verification

---

## 🏗️ Current Development Architecture

The following diagram illustrates the multi-modal diagnostic flow of the FusionMedAI framework, from raw heterogeneous ingestion to multi-level predictive fusion:

![System Architecture](docs/architecture_v1.png)

*Figure 1: High-level architectural overview of the FusionMedAI multi-modal pipeline.*

---

## 🛠️ Tech Stack

- Python 3.12
- PyTorch
- Torchvision
- OpenCV
- Albumentations
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorBoard

---

## 🎯 Project Goals

- **Multi-Modal Diagnostic Fusion**: Integrate clinical eye scans, demographic records, diabetic lab markers, and foot ulcer images into unified disease staging.
- **Academic-Grade Reproducibility**: Enforce strict data validation, deterministic stratified splitting, and reproducible pipelines.
- **Fail-Fast Clinical Engineering**: Ensure dataset integrity (e.g., shape, resolution, aspect ratio, label bounds, file corruption) is programmatically verified before training begins.
- **Interpretability & Trust**: Build transparent models using Explainable AI (XAI) tools like SHAP and Integrated Gradients to support clinical decision-making.

---

## 📌 Current Implementation

Implemented:
- ✔ Dataset Preparation
- ✔ Metadata Generation
- ✔ Dataset Verification
- ✔ Stratified Splitting
- ✔ RetinaDataset
- ✔ Image Transforms
- ✔ DataLoader
- ✔ End-to-End Verification
- ✔ Exploratory Data Analysis
- ✔ EfficientNet-B0 Baseline Framework
- ✔ Modular Training Pipeline
- ✔ Checkpoint Management
- ✔ TensorBoard Integration
- ✔ Experiment Versioning
- ✔ Standalone Inference
- ✔ Framework Verification

In Progress:
- 🔄 Baseline Training & Hyperparameter Optimization

---

## 🚀 Releases

| Version | Status |
|---------|--------|
| v0.1.0  | Dataset Preparation ✅ |
| v0.2.0  | Data Pipeline ✅ |
| v0.3.0  | Exploratory Data Analysis ✅ |
| v0.4.0  | Baseline Model Framework ✅ |
| v0.5.0  | Baseline Training & Experiments 🔄 |
| v0.6.0  | Explainability |
| v1.0.0  | Retina Module |

---

## 📊 Project Status & Progress

The framework is organized into specialized domain modules. The current status of development is tracked below:

### Overall Progress

| Module | Core Features | Status |
| :--- | :--- | :--- |
| **Retina Module** | Diabetic Retinopathy staging (APTOS 2019 / IDRiD) | 🔄 In Progress |
| **Foot Ulcer Module** | Wound segmentation and infection classification | ⬜ Not Started |
| **Clinical Module** | Tabular EHR risk prediction and feature extraction | ⬜ Not Started |
| **Fusion Engine (ACARA-U)** | Joint embedding & uncertainty-weighted cross-attention | ⬜ Not Started |

### Retina Module Development Progress

- [x] Baseline Model Framework
- [x] Modular Training Pipeline
- [x] Experiment Tracking
- [x] Checkpoint Manager
- [x] Inference Module
- [x] Verification Framework
- [ ] Baseline Training
- [ ] Hyperparameter Optimization
- [ ] Architecture Comparison
- [ ] Explainability
- [ ] Calibration
- [ ] External Evaluation

---

## 📁 Repository Structure

```directory
FusionMedAI/
├── datasets/                 # Labeled medical databases
│   ├── raw/                  # Unmodified raw clinical inputs (e.g., aptos2019)
│   ├── interim/              # Generated diagnostic reports & logs
│   └── processed/            # Final versioned data splits
├── docs/                     # Architectural diagrams & specifications
├── experiments/              # Isolated model checkpoints and run artifacts
├── notebooks/                # Academic Jupyter notebooks
│   └── retina/
│       ├── eda.ipynb         # Interactive dataset exploratory analysis
│       ├── extract_stats.py  # Concurrent statistical feature extractor
│       └── run_eda_analysis.py # Automated EDA & report generation engine
├── research/                 # Academic notebooks & clinical engineering logs
│   ├── Volume_01_Dataset_Preparation/  # Raw audits, checks, and metadata logic
│   ├── Volume_02_Data_Pipeline/        # Data flow, complexities, and pipeline decisions
│   ├── Volume_03_Exploratory_Data_Analysis/ # Spatial, RGB, quality, and duplicate analysis
│   └── Volume_04_Baseline_Model/       # EfficientNet-B0 backbone, trainer design, checkpointing
├── src/                      # Production source codebase
│   ├── config.py             # Centralized pipeline configuration
│   ├── inference.py          # Standalone inference API (single & batch)
│   ├── data/                 # Data loading, transforms, and splits
│   ├── models/               # Model definitions, wrappers, factory
│   ├── training/             # Trainer class, optimizers, schedulers
│   └── utils/                # Helper utilities (seeds, device, files)
├── verification/             # Independent verification scripts
│   ├── data/                 # Ingestion & split checks
│   └── model/                # Model, checkpoint, and training dry-run verifications
├── LICENSE                   # Open-source licensing
└── requirements.txt          # Virtual environment dependencies
```

---

## 📚 Research Documentation

Each engineering phase is documented in detail:
- **Volume 01 — Dataset Preparation** (located at [research/Volume_01_Dataset_Preparation/](research/Volume_01_Dataset_Preparation/))
- **Volume 02 — Data Pipeline** (located at [research/Volume_02_Data_Pipeline/](research/Volume_02_Data_Pipeline/))
- **Volume 03 — Exploratory Data Analysis** (located at [research/Volume_03_Exploratory_Data_Analysis/](research/Volume_03_Exploratory_Data_Analysis/))
- **Volume 04 — Baseline Model Development** (located at [research/Volume_04_Baseline_Model/](research/Volume_04_Baseline_Model/))

Each volume contains:
- Introduction & background
- Objectives & contributions
- Technical design decisions & trade-offs
- Verification methodology & coverage
- Multi-dimensional figures and dashboards (for Volume 3)
- Research limitations & mitigations
- Concluding roadmaps

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
> The APTOS 2019 dataset is not distributed with this repository due to Kaggle licensing. Download it separately and place it under `datasets/raw/aptos2019/`.

Organize the files into the following directory layout:

```directory
datasets/
└── raw/
    └── aptos2019/
        ├── train.csv
        └── train_images/
            ├── 000c1434d8d7.png
            ├── 001639a39701.png
            └── ...
```

Run the pipeline setup and verification scripts in order:

```bash
# Step 1: Run raw dataset checks & verify images
python src/data/verify_dataset.py

# Step 2: Generate dataset metadata
python src/data/generate_metadata.py

# Step 3: Compute stratified train/validation/test splits
python src/data/split_dataset.py

# Step 4: Execute end-to-end pipeline verification
python src/data/verify_pipeline.py

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

Before launching full GPU training jobs, the baseline Retina model framework was validated using CPU dry-runs:

- **Model verification**: Successful initialization of EfficientNet-B0, parameter count profile (4,013,953 parameters, ~15.31 MB), forward check, and feature extraction mapping check (`[B, 1280, 7, 7]`).
- **Training verification**: Backpropagation verified (checking optimizer step updates parameters), loss check, and learning rate decay validation (decaying from $1 \times 10^{-4}$ to $9.1 \times 10^{-5}$ on Step 1).
- **Checkpoint verification**: verified state dictionary matching for parameters, history logs, optimizer state, scheduler state, and exact value reconstruction under `allclose` check.
- **Inference verification**: Standalone inference check on single-image and batch data loaders, including confidence scoring and processing latency checks (~35 ms/image on CPU).

The framework is now ready for full-scale GPU training and experimental evaluation.

---

## 🗺️ Project Roadmap

- **v0.1.0 (Dataset Preparation)**: Completed raw audit, metadata generation, and resolution scanning. ✅
- **v0.2.0 (Data Pipeline)**: Completed stratified split, lazy loading, transforms, and E2E verification. ✅
- **v0.3.0 (Exploratory Data Analysis)**: Completed concurrent stats extraction, RGB profiling, duplicate audit, quality scoring, and automated reports. ✅
- **v0.4.0 (Baseline Framework)**: Built custom model wrapper, factory, BaseClassifier, trainer, mixed precision (AMP), Early Stopping, checkpointing, standalone inference, and verification framework. ✅
- **v0.5.0 (Baseline Training & Experiments)**: baseline training, hyperparameter optimization, loss comparisons (Focal, Weighted CE), optimizer and scheduler comparisons. 🔄
- **v0.6.0 (Explainability & Calibration)**: Grad-CAM post-hoc visual explanation, calibration curves, and uncertainty estimation.
- **v1.0.0 (Retina Module)**: Production release of explainable, calibrated Retina module.
- **v2.0.0 (Foot Ulcer Module Complete)**: Integrate wound segmentation models.
- **v3.0.0 (Clinical Module Complete)**: Integrate EHR structured features and classification networks.
- **v4.0.0 (FusionMedAI Complete)**: Release unified multi-modal ACARA-U Fusion model.
## ⚖️ License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
