# 6. Discussion

Post-hoc calibration via Temperature Scaling has significantly improved the reliability of our base model's predictions. The model was initially overconfident, but the learned temperature scalar successfully softened the output distribution.

## Why Medical AI Needs Calibration
In safety-critical domains like medical imaging and diagnostic AI, raw accuracy is insufficient. A model that predicts a disease with 99% confidence must be correct exactly 99% of the time. If the model is overconfident, a physician might act on an incorrect high-confidence prediction, leading to inappropriate treatments. Calibration ensures that the model "knows what it doesn't know," allowing human clinicians to trust its probabilistic outputs.

## Why Temperature Scaling
Temperature Scaling was selected because it is computationally efficient, strictly preserves model accuracy, and is widely regarded as an effective baseline for deep neural networks. Alternative methods like Vector Scaling or Isotonic Regression might achieve slightly better empirical metrics but are substantially more prone to overfitting on limited validation data, adding unnecessary complexity.

## The Benefit of Post-Hoc Processing (No Retraining)
Applying calibration as a post-hoc step decouples model training from model scaling. This is beneficial because the primary neural architecture (EfficientNet-B3) could be trained to maximize discriminatory power without being constrained by calibration penalties. The calibration step requires just a few seconds of L-BFGS optimization on frozen logits, entirely removing the computational burden of retraining a deep network.

## Preparing for Uncertainty Estimation (Step 8)
The calibrated probabilities are critical for the downstream tasks in Volume 08 (Uncertainty Estimation). Reliable base probabilities ensure that entropy-based uncertainty metrics or evidential fusion systems (like ACARA-U) are operating on mathematically sound foundations rather than distorted, overconfident signals.
