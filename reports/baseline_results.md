# FusionMedAI: Dry-Run Framework Verification Report

> [!IMPORTANT]
> This report summarizes the results of a framework verification dry-run executed on CPU.
> The reported classification metrics do **not** represent the final performance of EfficientNet-B0.
> Their purpose is to verify that the complete training, evaluation, checkpointing, and inference pipelines execute correctly.
>
> Official baseline performance will be reported after full training during Step 5.

## Experiment Run Summary
* **Run Version**: `v001_efficientnet_b0`
* **Execution Mode**: `Dry Run`
* **Dataset**: `Verification Subset`
* **Model Backbone**: `efficientnet_b0`
* **Evaluation Timestamp**: `2026-06-28 19:53:44`
* **Target Device**: `cpu`

## Model Profile
* **Total Parameters**: 4,013,953
* **Trainable Parameters**: 4,013,953
* **Model Size**: 15.31 MB

## Framework Verification Status
| Component | Status |
| :--- | :---: |
| Model Loading | ✅ |
| Forward Pass | ✅ |
| Backpropagation | ✅ |
| Optimizer | ✅ |
| Scheduler | ✅ |
| Metrics | ✅ |
| Checkpointing | ✅ |
| Inference | ✅ |

## Dry-Run Evaluation Metrics
| Metric | Value | Purpose / Description |
| :--- | :--- | :--- |
| **Quadratic Weighted Kappa (QWK)** | **0.0000** | Primary performance metric for ordinal classification. |
| **Macro F1-Score** | 0.0632 | Balanced harmonic mean across all DR classes. |
| **Accuracy** | 0.1875 | Overall ratio of correct predictions. |
| **Balanced Accuracy** | 0.2000 | Accuracy adjusted for class imbalances. |
| **Sensitivity (Recall)** | 0.2000 | True positive rate averaged macro-wise. |
| **Specificity** | 0.8000 | True negative rate averaged macro-wise. |
| **Precision** | 0.0375 | Positive predictive value averaged macro-wise. |
| **ROC AUC (Macro)** | 0.5000 | Area under the Receiver Operating Characteristic curve. |

## Inference Speed / Latency Profile
* **Average Inference Speed**: `17.71 ms/image`
* **Throughput**: `56.48 FPS`

## Verification Confusion Matrix (Counts)
| True \ Predicted | No DR (0) | Mild (1) | Moderate (2) | Severe (3) | Proliferative (4) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Class 0 | 0 | 0 | 0 | 9 | 0 |
| Class 1 | 0 | 0 | 0 | 5 | 0 |
| Class 2 | 0 | 0 | 0 | 10 | 0 |
| Class 3 | 0 | 0 | 0 | 6 | 0 |
| Class 4 | 0 | 0 | 0 | 2 | 0 |

## Visualization Files Generated
* Training History Curves: `experiments/v001_efficientnet_b0/curves/`
* Confusion Matrix: `experiments/v001_efficientnet_b0/confusion_matrix/confusion_matrix.png`
* ROC Curves: `experiments/v001_efficientnet_b0/roc/roc_curve.png`

## Environment
* **Python Version**: `3.12.2`
* **PyTorch Version**: `2.11.0+cpu`
* **Torchvision Version**: `0.26.0+cpu`
* **Random Seed**: `42`
* **Experiment Folder**: `experiments/v001_efficientnet_b0/`

## Note
The metrics presented in this report originate from a CPU-based verification execution intended to validate the correctness of the training framework.
These values should not be interpreted as the expected performance of the EfficientNet-B0 baseline.
Final benchmark results will be generated after full-scale training and hyperparameter optimization during Step 5.
