# Volume 04: Baseline Framework

## Purpose

Develop a modular, reproducible deep learning framework that serves as the engineering foundation for the FusionMedAI Retina Module.

## Outcome

- ✓ EfficientNet-B0 baseline implementation
- ✓ Modular PyTorch training framework
- ✓ Checkpoint management and experiment versioning
- ✓ Comprehensive evaluation pipeline
- ✓ Framework verification and reproducibility

The framework established in this volume subsequently served as the common infrastructure for all Retina Module development, including architecture benchmarking, explainability, probability calibration, and future uncertainty estimation.

## Completed Downstream Volumes

The baseline training and evaluation infrastructure established in this volume was reused throughout the completed Retina research stages:

- ✓ Volume 05 – Architecture Benchmarking
- ✓ Volume 06 – Model Explainability
- ✓ Volume 07 – Probability Calibration
- ✓ Volume 08 – Prediction Uncertainty Estimation
- ✓ Volume 09 – Retina Module Integration & Finalization

## Current Status

**Volume 04 – Baseline Framework: ✅ Complete**

The baseline training framework is complete and frozen for the Retina Module. Its modular design allowed other backbones to be benchmarked seamlessly, and it remains the design reference for the upcoming modules.

The next development phase is the **Foot Ulcer Module**, which will adapt this baseline training and checkpointing framework for the classification of foot ulcer wound images.