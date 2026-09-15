# Chapter 01 — Objectives & Research Position

## 1.1 Core Research Objective

Modern deep neural networks trained with cross-entropy loss often produce miscalibrated probability predictions, where high softmax confidences ($P(y=c|X) > 0.90$) significantly exceed the actual observed empirical classification accuracy.

Phase 10.7 addresses this problem for the frozen Phase 10.5 primary Foot Ulcer model (**EfficientNet-B3**). 

The primary research objective is:

> **Determine whether raw EfficientNet-B3 softmax probabilities are miscalibrated, fit post-hoc probability calibrators (Temperature Scaling and Vector Scaling) exclusively on validation-set logits, and measure whether calibration improves probabilistic reliability (NLL, ECE, MCE, Brier score) on the held-out test set.**

---

## 1.2 Strict Model & Split Freezing Rules

To ensure strict scientific validity and prevent data leakage:

1. **Model Weights Frozen**: EfficientNet-B3 parameters from `experiments/foot/architecture_benchmark/efficientnet_b3/checkpoints/best_model.pt` are frozen. No retraining or fine-tuning of network weights occurs.
2. **Strict Validation-Set Fitting**: All calibration parameters ($T^*$ for Temperature Scaling; $W^*, b^*$ for Vector Scaling) are optimized **exclusively on validation logits** ($N = 1,006$).
3. **Unbiased Held-Out Test Evaluation**: The test split ($N = 1,006$) is evaluated **only after calibration parameters have been frozen**. The test set is never used to select or tune calibration parameters.

---

## 1.3 Hypotheses & Evaluation

- **$H_1$ (Uncalibrated Miscalibration)**: Raw softmax probabilities will exhibit non-zero calibration error ($ECE > 0.04$).
- **$H_2$ (Likelihood & Calibration Improvement)**: Post-hoc calibration will reduce validation and test NLL and ECE relative to uncalibrated raw softmax probabilities.
- **$H_3$ (Argmax Rank Preservation)**: Post-hoc calibration will preserve raw prediction rankings with minimal classification metric change ($\Delta \le 0.001$).

### Empirical Hypothesis Outcome
- **$H_1$**: **SUPPORTED**. Raw baseline exhibits $ECE = 0.0424$ on the held-out test set.
- **$H_2$**: **SUPPORTED**. Vector Scaling reduces held-out test ECE by **26.18%** (from $0.0424$ to $0.0313$) and NLL from $0.8779$ to $0.8749$.
- **$H_3$**: **PARTIALLY SUPPORTED**. Temperature Scaling satisfied $H_3$ strictly ($\Delta = 0.0000$). However, the selected **Vector Scaling** calibrator altered argmax class rankings for a small fraction of boundary samples, resulting in a Macro F1 increase of **+0.0075** ($0.6683 \to 0.6758$) and accuracy increase of **+0.0079** ($0.6690 \to 0.6769$). This classification metric gain is reported descriptively and was not used as a selection criterion.
