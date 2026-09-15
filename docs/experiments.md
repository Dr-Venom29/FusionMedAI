# Experiments Documentation

## Overview

FusionMedAI stores each experiment with its configuration, training history, checkpoints, predictions, and evaluation results.

Each experiment is executed within an isolated versioned directory containing its complete configuration, checkpoints, logs, predictions, and evaluation artifacts.

---

# Experiment Structure

Each experiment creates a dedicated directory under its modality:

```directory
experiments/
├── retina/
└── foot/
    ├── architecture_benchmark/
    ├── calibration/
    ├── explainability/
    └── final_model/
```

This organization prevents experiment outputs from being overwritten and enables direct comparison between runs.

---

# Experiment Tracking

For every experiment, the framework records:

* Model architecture
* Hyperparameters
* Optimizer configuration
* Learning rate scheduler
* Random seed
* Training history
* Evaluation metrics
* Checkpoints
* Prediction logs

These records provide the configuration and artifacts required to reproduce and inspect an experiment.

---

# Experiment Logging

The framework maintains:

* `training_history.csv` — Epoch-by-epoch training and validation loss/metrics.
* `training_summary.json` — High-level training execution summary and convergence stats.
* `error_analysis.csv` — Prediction-level misclassification logs with class probabilities and confidence scores.
* `test_evaluation.json` / `test_evaluation.md` — Frozen test set evaluation benchmarks.

---

# Experiment Categories by Modality

## Retina Module Experiments — Completed

### Backbone Architectures
* EfficientNet-B0
* EfficientNet-B3 (Selected Retina Backbone)
* ConvNeXt-Tiny
* Swin-Tiny
* ViT-B/16

### Calibration
* Temperature Scaling (Post-hoc calibration on the validation split)

### Uncertainty
* Monte Carlo Dropout (Stochastic ensembling with $N=25$ passes on the test split)

---

## Foot Ulcer Module Experiments

### Baseline Experiments — Completed
* ResNet-50 Baseline
  - Macro F1: `0.6339`
  - Accuracy: `0.6372`
  - Balanced Accuracy: `0.6391`

### Architecture Benchmarking — Completed
* EfficientNet-B0
* EfficientNet-B3 — Selected Foot Ulcer Backbone
* ConvNeXt-Tiny
* Swin-Tiny
* ViT-B/16

### Explainability — Completed
* Grad-CAM
* Attribution sanity checks

### Probability Calibration — Completed
* Temperature Scaling
* Vector Scaling — Selected

### Uncertainty Estimation
* Pending

---

## Possible Future Experiments

Future studies may evaluate:

### Hyperparameters
* Learning rate schedule tuning
* Batch size variation
* Weight decay sensitivity

### Loss Functions
* Weighted CrossEntropy (Sqrt Inverse Frequency)
* Focal Loss
* Label Smoothing CrossEntropy

---

# Evaluation Criteria

Every experiment is evaluated using standard metrics to ensure consistent comparison:

* Macro F1-score (Primary Metric)
* Top-1 Accuracy
* Balanced Accuracy
* Weighted F1-score
* Class-wise Precision, Recall, F1
* Multi-class Confusion Matrix
* Macro ROC-AUC
* Model parameter count and latency

---

# Design Principles

The experiment management framework emphasizes:

* Reproducibility
* Version control
* Modular experimentation
* Traceability

This structure supports comparison of preprocessing, optimization, and model architecture experiments.
