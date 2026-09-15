# Chapter 04 — Raw Model Calibration Diagnostics

## 4.1 Raw Softmax Baseline Evaluation

The uncalibrated raw softmax probabilities produced by the frozen Phase 10.5 **EfficientNet-B3** checkpoint (`best_model.pt`) were evaluated across both validation ($N=1,006$) and held-out test ($N=1,006$) splits under the fixed 10 equal-width binning protocol.

### Raw Baseline Performance Metrics

| Split | NLL ↓ | ECE ↓ | MCE ↓ | Brier ↓ | Accuracy | Mean Conf | Overconf Gap | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation ($N=1,006$)** | 0.8692 | 0.0541 | 0.1412 | 0.1130 | 0.6720 | 0.6657 | -0.0063 | 0.6705 |
| **Test ($N=1,006$)** | 0.8779 | 0.0424 | 0.0753 | 0.1155 | 0.6690 | 0.6627 | -0.0063 | 0.6683 |

---

## 4.2 Baseline Diagnostic Observations

- **Expected Calibration Error (ECE)**: The uncalibrated test ECE is **0.0424**, indicating miscalibration across confidence bins.
- **Brier Score**: The raw multi-class Brier score (MSE between predicted probabilities and one-hot ground truth labels) is **0.1155**.
- **Negative Log-Likelihood (NLL)**: The raw cross-entropy loss is **0.8779**.

---

## 4.3 Motivation for Post-Hoc Calibration

In medical AI applications like diabetic foot ulcer assessment, uncalibrated probabilities present clinical risks:
1. Miscalibrated confidence outputs can prevent secondary clinical review.
2. Inconsistent confidence scores undermine trust in automated triage tools.
3. Downstream multi-modal fusion layers (e.g., ACARA-U) rely on calibrated probabilities for uncertainty-weighted decision fusion.

Post-hoc calibration is strictly required to adjust probability scales without altering model feature representations.
