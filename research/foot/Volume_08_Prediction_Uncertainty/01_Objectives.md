# Chapter 01 — Objectives & Research Position

## 1. Context & Motivation

Diabetic Foot Ulcer (DFU) Wagner grading involves clinical ambiguity, particularly across adjacent ulcer severity stages (e.g., Grade 2 deep ulcer vs Grade 3 deep ulcer with osteomyelitis or abscess). Standard neural network softmax outputs suffer from overconfidence after calibration or scaling if data instance noise or ambiguous visual features are present.

Phase 10.8 establishes a principled, post-hoc prediction uncertainty estimation framework for the primary Foot Ulcer model (**EfficientNet-B3**) without altering model parameters or refitting probability calibrators.

---

## 2. Strict Freeze Requirements

To prevent data leakage and experimental drift:
1. **Model Weights Frozen**: `experiments/foot/architecture_benchmark/efficientnet_b3/checkpoints/best_model.pt` is strictly frozen. No backpropagation or weight retraining is conducted.
2. **Calibrator Parameters Frozen**: `experiments/foot/final_model/calibration.json` (Vector Scaling parameters $w^* = [1.0410, 1.0430, 0.8711, 1.1301], b^* = [0.0292, 0.0625, 0.0630, -0.1547]$) are strictly frozen.
3. **Data Splits Frozen**: Validation set ($N=1,006$) is used exclusively for stochastic pass count convergence analysis; held-out test set ($N=1,006$) is evaluated strictly once for final reporting.

---

## 3. Core Research Questions

1. **Pass Count Convergence**: How many stochastic MC Dropout passes ($N$) are required for predictive entropy and variance estimates to reach numerical stability ($\Delta H < 10^{-3}$)?
2. **Error Detection Capability**: Can estimated uncertainty metrics (Predictive Variance, Total Predictive Entropy, Mutual Information) effectively discriminate between correct and incorrect predictions on held-out test images?
3. **Selective Prediction & Risk-Coverage**: Does rejecting high-uncertainty samples monotonically reduce the risk (error rate) on remaining clinical predictions?
4. **Classwise Difficulty & Boundary Disaggregation**: How does uncertainty vary across Wagner grades (G1, G2, G3, G4), and does it reflect boundary ambiguity between Grade 2 and Grade 3?
5. **Uncertainty-Explainability Connection**: Does high prediction uncertainty correlate with diffuse or multi-focal attribution patterns in Grad-CAM feature heatmaps?

---

## 4. Methodological Alignment

The Foot Ulcer uncertainty estimation methodology maintains research consistency with the Retina module while operating independently on 4-class Wagner classification data.
