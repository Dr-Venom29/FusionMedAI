# FusionMedAI: Baseline Model Evaluation Report

## Experiment Run Summary
* **Run Version**: `v001`
* **Run Description**: `efficientnet_b0`
* **Model Backbone**: `efficientnet_b0`
* **Evaluation Timestamp**: `2026-06-28 14:25:13`
* **Target Device**: `cpu`

## Model Profile
* **Total Parameters**: 4,013,953
* **Trainable Parameters**: 4,013,953
* **Model Size**: 15.31 MB

## Overall Performance Metrics
| Metric | Value | Purpose / Description |
| :--- | :--- | :--- |
| **Quadratic Weighted Kappa (QWK)** | **-0.2129** | Primary performance metric for ordinal classification. |
| **Macro F1-Score** | 0.0429 | Balanced harmonic mean across all DR classes. |
| **Accuracy** | 0.0938 | Overall ratio of correct predictions. |
| **Balanced Accuracy** | 0.1000 | Accuracy adjusted for class imbalances. |
| **Sensitivity (Recall)** | 0.1000 | True positive rate averaged macro-wise. |
| **Specificity** | 0.7665 | True negative rate averaged macro-wise. |
| **Precision** | 0.0273 | Positive predictive value averaged macro-wise. |
| **ROC AUC (Macro)** | 0.5029 | Area under the Receiver Operating Characteristic curve. |

## Inference Speed / Latency Profile
* **Average Inference Speed**: `17.41 ms/image`
* **Throughput**: `57.43 FPS`

## Confusion Matrix (Counts)
| True \ Predicted | No DR (0) | Mild (1) | Moderate (2) | Severe (3) | Proliferative (4) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Class 0 | 0 | 0 | 0 | 9 | 0 |
| Class 1 | 2 | 0 | 0 | 3 | 0 |
| Class 2 | 5 | 0 | 0 | 5 | 0 |
| Class 3 | 2 | 0 | 1 | 3 | 0 |
| Class 4 | 0 | 0 | 0 | 2 | 0 |

## Visualization Files Saved
* Training History Curves: `experiments/v001_efficientnet_b0/curves/`
* Confusion Matrix: `experiments/v001_efficientnet_b0/confusion_matrix/confusion_matrix.png`
* ROC Curves: `experiments/v001_efficientnet_b0/roc/roc_curve.png`
