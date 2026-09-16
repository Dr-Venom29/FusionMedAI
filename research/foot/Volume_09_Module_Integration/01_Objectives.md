# Chapter 01 — Phase 10.9 Objectives & Frozen Asset Specifications

## 1. Primary Objective

The objective of Phase 10.9 is to construct a unified, self-contained inference interface (**`FootModule`**) for Diabetic Foot Ulcer (DFU) Wagner classification (`src/foot/foot_module.py`), unifying:
1. Deterministic inference via the frozen **EfficientNet-B3** primary classifier.
2. Post-hoc probability calibration via frozen **Vector Scaling** parameters ($w^*, b^*$).
3. Stochastic uncertainty estimation via **MC Dropout** ($N^*=10$ passes, Option B pipeline).
4. Deterministic spatial explainability via **Grad-CAM** attributions.

The module interface is designed to achieve contract parity with the existing `RetinaModule`, establishing a uniform modality interface across FusionMedAI before multimodal fusion layer integration.

---

## 2. Frozen Artifact Specification

During module integration, all upstream model checkpoints, calibrator weights, and uncertainty pass-count hyper-parameters remain strictly frozen. No retraining or re-fitting occurs.

| Asset Layer | Component Name | Source Artifact Path | Frozen Parameters / Configuration |
| :--- | :--- | :--- | :--- |
| **Model Classifier** | EfficientNet-B3 | `experiments/foot/architecture_benchmark/efficientnet_b3/checkpoints/best_model.pt` | Frozen weights (Macro F1 = 0.6683, Macro ROC-AUC = 0.8685) |
| **Probability Calibrator** | Vector Scaling | `experiments/foot/final_model/calibration.json` | $w^* = [1.0410, 1.0430, 0.8711, 1.1301]$, $b^* = [0.0292, 0.0625, 0.0630, -0.1547]$ (ECE = 0.0313) |
| **Uncertainty Estimator** | MC Dropout (Option B) | `experiments/foot/final_model/uncertainty.json` | $N^* = 10$ passes, $\text{Dropout} \rightarrow \text{train()}$, $\text{BatchNorm} \rightarrow \text{eval()}$ |
| **Explainability** | FootGradCAM | Target layer `backbone.features.8` | Deterministic forward-backward hook extraction under `model.eval()` |

---

## 3. Scope & Operational Constraints

1. **Non-Causal & Non-Clinical Scope**: The integrated `FootModule` provides automated multi-class score probabilities, confidence, uncertainty metrics, and attribution heatmaps. Outputs evaluate empirical model behavior on benchmark scans and do not constitute clinical diagnostic claims or autonomous surgical recommendations.
2. **Inference Modes**: Supports both standard fast inference (`generate_cam=False`, skipping heatmap tensor creation) and full explainable inference (`generate_cam=True`).
3. **Input Validation Isolation**: Malformed, missing, or corrupt image files are intercepted at the boundary layer before tensor allocation, returning explicit diagnostic exceptions (`FileNotFoundError`, `ValueError`, `TypeError`).
