# Experiments Documentation

## Overview

FusionMedAI follows a structured experiment management strategy to ensure reproducibility, fair model comparison, and systematic evaluation.

Each experiment is executed within an isolated versioned directory containing its complete configuration, checkpoints, logs, predictions, and evaluation artifacts.

---

# Experiment Structure

Each experiment automatically creates a dedicated directory:

```directory
experiments/
└── v001_efficientnet_b0/
    ├── config.json
    ├── history.csv
    ├── history.json
    ├── predictions.csv
    ├── baseline_results.csv
    ├── checkpoints/
    ├── curves/
    ├── confusion_matrix/
    ├── roc/
    └── tensorboard/
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
* TensorBoard logs

This information enables exact reproduction of experimental results.

---

# Experiment Logging

The framework maintains:

* `experiment_log.xlsx` — Global experiment registry.
* `baseline_results.csv` — Summary of completed baseline experiments.
* `history.csv` / `history.json` — Epoch-by-epoch training history.
* `predictions.csv` — Test predictions for downstream analysis.

---

# Current Status

Completed:

* Experiment versioning
* Checkpoint management
* TensorBoard integration
* Configuration export
* History logging
* Prediction export

Planned:

* Full baseline training
* Hyperparameter optimization
* Learning rate experiments
* Loss function comparison
* Optimizer comparison
* Architecture benchmarking

---

# Planned Experiment Categories

Future studies will evaluate:

## Baseline Model

* EfficientNet-B0

## Hyperparameters

* Learning rate
* Batch size
* Weight decay
* Number of epochs

## Loss Functions

* Cross Entropy
* Weighted Cross Entropy
* Focal Loss
* Ordinal Loss

## Optimizers

* AdamW
* Adam
* SGD with Momentum

## Learning Rate Schedulers

* Cosine Annealing
* StepLR
* OneCycleLR
* ReduceLROnPlateau

## Backbone Architectures

* EfficientNet-B0
* EfficientNet-B3
* ConvNeXt
* Swin Transformer
* Vision Transformer

---

# Evaluation Criteria

Every experiment will be evaluated using the same metrics to ensure fair comparison:

* Quadratic Weighted Kappa (QWK)
* Macro F1-score
* Balanced Accuracy
* Accuracy
* Precision
* Recall
* Specificity
* ROC-AUC
* Inference latency
* Model size

---

# Design Principles

The experiment management framework emphasizes:

* Reproducibility
* Version control
* Modular experimentation
* Fair benchmarking
* Traceability
* Scalability

This infrastructure enables systematic comparison of preprocessing strategies, optimization methods, and model architectures throughout the FusionMedAI project.
