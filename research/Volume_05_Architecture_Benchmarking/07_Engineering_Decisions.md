# Chapter 7: Engineering Decisions

## 7.1 High-Level Architecture Design
To guarantee fairness and reproducibility, the following software engineering decisions were enforced:
- **Model-Agnostic Framework**: The training and evaluation loops were entirely decoupled from the model architecture through a unified `BaseClassifier` interface.
- **Configuration-Driven Benchmarking**: A single `benchmark_config.json` drove the entire execution loop, allowing automated sequential training of all models without manual intervention.
- **Frozen Benchmark Configuration**: Hyperparameters were strictly locked across all experiments to prevent configuration drift.
- **Deterministic Execution**: Seed `42` was enforced at the Python, NumPy, and PyTorch levels to ensure that benchmark results could be perfectly replicated.
- **Automated experiment archiving**: Ensured each benchmark execution preserved checkpoints, logs, configuration files, and generated artifacts for later explainability and calibration stages.
- **Fixed Backbone Foundation**: The selected EfficientNet-B3 checkpoint served as the fixed foundation for all downstream Retina Module development.
