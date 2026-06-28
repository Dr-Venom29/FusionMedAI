# Chapter 11: Engineering Decisions

Several architectural and software engineering decisions were made during the development of the baseline framework to maximize modularity, reproducibility, maintainability, and future extensibility.

## 1. Torchvision over timm

The baseline EfficientNet-B0 implementation uses the official `torchvision.models` implementation rather than third-party libraries. This minimizes external dependencies, ensures long-term compatibility with the PyTorch ecosystem, and simplifies deployment.

Future backbone architectures such as ConvNeXt, Swin Transformer, and Vision Transformer may be integrated through `timm` where official implementations are unavailable or less mature.

---

## 2. Common BaseClassifier Interface

A shared `BaseClassifier` abstraction was introduced to standardize all backbone architectures.

Every model exposes a consistent interface for:

* Forward inference
* Feature extraction
* Parameter profiling

This enables future architectures to be integrated into the same training framework without modifying the trainer, evaluation pipeline, or inference engine.

---

## 3. Model Factory Pattern

Rather than instantiating architectures directly inside the training pipeline, all models are created through a centralized model factory.

This design:

* Decouples model selection from training.
* Simplifies architecture comparison.
* Enables configuration-driven experimentation.

---

## 4. Modular Training Pipeline

The training framework separates responsibilities into dedicated modules:

* Training
* Validation
* Testing
* Metrics
* Optimization
* Scheduling
* Checkpointing
* Visualization

This modular design improves readability, maintainability, and experimental flexibility while reducing code duplication.

---

## 5. Versioned Experiments

Every experiment executes within its own isolated directory (e.g., `experiments/v001_efficientnet_b0/`).

Each run stores:

* Configuration
* Checkpoints
* Training history
* TensorBoard logs
* Predictions
* Evaluation outputs

This prevents accidental overwriting of previous experiments and guarantees reproducibility.

---

## 6. QWK-Based Checkpoint Selection

Instead of monitoring validation loss, the framework selects the best checkpoint using **Quadratic Weighted Kappa (QWK)**.

Since Diabetic Retinopathy grading is an ordinal classification problem, QWK provides a clinically more meaningful optimization objective than conventional accuracy or loss.

---

## 7. Automatic Mixed Precision (AMP)

Training incorporates PyTorch Automatic Mixed Precision using `torch.amp.autocast` together with `GradScaler`.

This design reduces computational cost while maintaining numerical stability during gradient updates.

---

## 8. Dedicated Verification Framework

Verification scripts were intentionally separated from the runtime source code and organized within a dedicated `verification/` directory.

Independent verification of:

* Dataset pipeline
* Model architecture
* Training loop
* Checkpoint management

ensures that infrastructure correctness is established before computationally expensive experiments begin.

---

## 9. Standalone Inference Module

Prediction logic is isolated from the training framework through a dedicated inference module supporting:

* Single-image inference
* Batch inference
* Confidence estimation
* Latency measurement

This separation prepares the framework for future deployment without modifying the training pipeline.

---

## 10. Lightweight Documentation Strategy

Automatically generated PDF reports were intentionally removed from the training pipeline.

Instead, Markdown reports are produced together with structured experiment logs and visualization outputs. PDF generation is deferred until thesis writing or publication preparation, reducing repository size and eliminating duplicate artifacts.

## Summary

These engineering decisions collectively establish a research-oriented framework that emphasizes modularity, reproducibility, scalability, and maintainability. The resulting architecture provides a stable foundation for subsequent hyperparameter optimization, backbone comparison, explainability, calibration, uncertainty estimation, and multimodal fusion within the FusionMedAI framework.

```mermaid
graph LR
    Modularity --> Reproducibility
    Reproducibility --> Scalability
    Scalability --> Maintainability
    Maintainability --> Extensibility
```
