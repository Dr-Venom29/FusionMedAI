# Volume 05: Architecture Benchmarking

This volume documents the systematic benchmarking of five state-of-the-art computer vision architectures under an identical experimental protocol to identify the optimal backbone for the FusionMedAI Retina Module.

## Objective

Evaluate CNN- and Transformer-based architectures using identical training configurations, preprocessing pipelines, and evaluation metrics to ensure a fair and reproducible comparison.

## Outcome

**Selected Backbone**

- **EfficientNet-B3**

**Completed Downstream Stages**

The selected EfficientNet-B3 backbone identified in this volume was utilized as the fixed foundation throughout the completed Retina stages:

- ✓ Volume 06 – Model Explainability
- ✓ Volume 07 – Probability Calibration
- ✓ Volume 08 – Prediction Uncertainty Estimation
- ✓ Volume 09 – Retina Module Integration & Finalization

**Current Status**

**Volume 05 – Architecture Benchmarking: ✅ Complete**

The benchmarking study is complete and frozen for the Retina Module. Its selection of EfficientNet-B3 establishes the backbone model used throughout all subsequent Retina development stages.

The next development phase is the **Foot Ulcer Module**, which will execute its own backbone architecture benchmarking to identify the optimal feature extractor for foot ulcer classification.

## Folder Organization

- `figures/` – Benchmark visualizations, including accuracy, QWK, ROC-AUC, performance vs. efficiency, latency, throughput, and other comparative plots.
- `01_...` through `09_...` – Detailed chapters covering the benchmarking protocol, selected architectures, training configuration, experimental results, engineering decisions, limitations, and final model selection.

## Summary

The benchmarking phase established **EfficientNet-B3** as the official Retina Module backbone by achieving the strongest balance between diagnostic performance and computational efficiency. This backbone serves as the fixed foundation for all subsequent Retina Module development, including explainability, calibration, uncertainty estimation, and final deployment.