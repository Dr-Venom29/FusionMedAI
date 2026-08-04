# Volume 05: Architecture Benchmarking

This volume documents the systematic benchmarking of five state-of-the-art computer vision architectures under an identical experimental protocol to identify the optimal backbone for the FusionMedAI Retina Module.

## Objective

Evaluate CNN- and Transformer-based architectures using identical training configurations, preprocessing pipelines, and evaluation metrics to ensure a fair and reproducible comparison.

## Outcome

**Selected Backbone**

- **EfficientNet-B3**

**Completed Downstream Stages**

- ✓ Volume 06 – Explainability (XAI)
- ✓ Volume 07 – Probability Calibration

**Upcoming Stages**

- → Volume 08 – Uncertainty Estimation
- → Retina Module Integration
- → Final Retina Module Evaluation

## Folder Organization

- `figures/` – Benchmark visualizations, including accuracy, QWK, ROC-AUC, performance vs. efficiency, latency, throughput, and other comparative plots.
- `tables/` – Exported CSV/XLSX benchmark results, metric rankings, and model profile summaries.
- `01_...` through `09_...` – Detailed chapters covering the benchmarking protocol, selected architectures, training configuration, experimental results, engineering decisions, limitations, and final model selection.

## Summary

The benchmarking phase established **EfficientNet-B3** as the official Retina Module backbone by achieving the strongest balance between diagnostic performance and computational efficiency. This backbone serves as the fixed foundation for all subsequent Retina Module development, including explainability, calibration, uncertainty estimation, and final deployment.