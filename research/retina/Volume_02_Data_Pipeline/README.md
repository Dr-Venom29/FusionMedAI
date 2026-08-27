# Volume 02: Data Pipeline

## Purpose

Develop a modular, reproducible, and verifiable data pipeline that transforms raw retinal images into standardized tensors for model training and evaluation.

## Outcome

- ✓ Stratified dataset splitting
- ✓ RetinaDataset implementation
- ✓ Modular transform pipelines
- ✓ PyTorch DataLoader framework
- ✓ End-to-end pipeline verification
- ✓ Reproducible preprocessing workflow

The standardized data pipeline established in this volume became the common data infrastructure for all subsequent Retina Module development.

## Adopted in Subsequent Volumes

The standardized data pipeline established in this volume was reused throughout the completed Retina research stages:

- ✓ Volume 03 – Exploratory Data Analysis
- ✓ Volume 04 – Baseline Framework
- ✓ Volume 05 – Architecture Benchmarking
- ✓ Volume 06 – Explainability (XAI)
- ✓ Volume 07 – Probability Calibration
- ✓ Volume 08 – Prediction Uncertainty Estimation
- ✓ Volume 09 – Retina Module Integration & Finalization

## Current Status

**Volume 02 – Data Pipeline: ✅ Complete**

The data pipeline is considered frozen for the Retina Module. Its verified components were reused throughout the subsequent Retina development stages and remain the foundation for future modality-specific pipelines.

The next development phase is the **Foot Ulcer Module**, which will establish its own dataset preparation and data pipeline following the same reproducibility and verification principles.