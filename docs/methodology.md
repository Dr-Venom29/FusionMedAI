# FusionMedAI Methodology

## Overview

FusionMedAI is a modular clinical intelligence framework designed for multi-modal medical diagnosis. The project follows a staged research methodology in which each module is independently developed, validated, and benchmarked before integration into the final multimodal fusion system.

This incremental approach improves reproducibility, simplifies experimentation, and enables fair evaluation of each component.

---

# Research Methodology

The development process follows the pipeline below:

```mermaid
flowchart TD
    DatasetPrep[Dataset Preparation]
    --> DataPipeline[Data Pipeline]
    --> EDA[Exploratory Data Analysis]
    --> BaselineFramework[Baseline Framework]
    --> ArchitectureBenchmarking[Architecture Benchmarking]
    --> Explainability
    --> Calibration
    --> UncertaintyEstimation[Uncertainty Estimation]
    --> ModuleCompletion[Module Completion]
    --> MultimodalFusion[ACARA-U Fusion]
```

Each stage is verified before progressing to the next stage.

---

# Module Independence

FusionMedAI consists of independent diagnostic modules:

* Retina Module
* Foot Ulcer Module
* Clinical Module

Each module performs:

* Dataset preparation
* Data preprocessing
* Model training
* Evaluation
* Explainability
* Probability Calibration
* Uncertainty Estimation

independently before multimodal integration.

---

# Dataset Alignment Statement

## Important Research Assumption

FusionMedAI **does not perform patient-level multimodal learning.**

The public datasets used throughout the project originate from different patient populations and therefore cannot be directly merged into a single patient-level dataset.

Consequently, patient identities are never assumed to correspond across datasets.

---

# Decision-Level Fusion (ACARA-U)

Instead of combining raw patient data, FusionMedAI adopts a **decision-level fusion** strategy.

Each module independently produces:

* Disease prediction
* Confidence score
* Reliability score
* Uncertainty estimate

These outputs are subsequently aggregated by the ACARA-U Fusion Engine to generate a unified clinical assessment.

This methodology avoids introducing artificial patient correspondences while maintaining methodological validity.

---

# Engineering Principles

The framework follows several core engineering principles:

* Modular software architecture
* Reproducible experimentation
* Configuration-driven execution
* Comprehensive verification
* Experiment versioning
* Clinically relevant evaluation metrics

---

# Current Project Status

Completed:
* Dataset Preparation
* Data Pipeline
* Exploratory Data Analysis
* Baseline Framework
* Architecture Benchmarking
* Explainability
* Probability Calibration
* Uncertainty Estimation

Current:
* Retina Module Integration

Planned:
* Clinical Module
* Foot Module
* ACARA-U Fusion

---

# Future Methodology

Once all individual modules have been validated, the final FusionMedAI methodology will integrate their outputs through the ACARA-U Fusion Engine using uncertainty-aware decision aggregation rather than feature-level patient fusion.

This approach preserves scientific validity while enabling multimodal clinical intelligence across heterogeneous public medical datasets.

---

# Architecture Benchmarking (Step 5)

FusionMedAI conducts rigorous fair-comparison benchmarking to identify the optimal architectural backbone for each module. The protocol ensures that inductive biases and architectural paradigms (e.g., CNNs vs Vision Transformers) are compared transparently without hyperparameter bias.

## Benchmarking Protocol
- **Frozen Environment**: All architectures are subjected to the exact same dataset, train/val/test splits, batch size, epochs, and random seeds.
- **Identical Optimization**: Confounding variables are eliminated by strictly fixing the optimizer (AdamW), learning rate scheduler (CosineAnnealingLR), and loss function (Weighted Cross-Entropy) across all models.
- **Hardware Efficiency Tracking**: Beyond diagnostic metrics (Accuracy, QWK, ROC-AUC), models are profiled for parameter count, FLOPs, MACs, peak VRAM, inference latency, and throughput.
- **Clinical Feasibility Selection**: The final model is selected based on a holistic assessment balancing high diagnostic power with edge-deployment feasibility (low memory footprint and low latency).

