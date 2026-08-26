# 2. Uncertainty Estimation Methods: Deterministic vs. Stochastic

Characterizing the uncertainty of deep learning systems involves selecting either single-pass deterministic approaches or multi-pass stochastic simulations. This chapter details the two paradigms, their theoretical motivations, and their practical trade-offs.

---

## 2.1 Deterministic Uncertainty Methods

Deterministic uncertainty relies on a single forward pass through a standard neural network. The uncertainty is extracted directly from the output probability distribution (softmax logits).

### 2.1.1 Methods Evaluated
1. **Maximum Softmax Probability (MSP)**:
   - The standard baseline for neural network confidence.
   - Requires post-hoc calibration (e.g., Temperature Scaling from Volume 7) to align logits with empirical accuracy.
2. **Predictive Entropy**:
   - Shannon entropy computed on the single-pass probabilities.
3. **Top-2 Margin**:
   - The gap between the first and second predicted classes.

### 2.1.2 Strengths
- **Computational Efficiency**: Requires exactly one forward pass ($O(1)$ complexity), introducing zero latency overhead.
- **Ease of Implementation**: Plugs directly into standard architectures without modifying model layers or training pipelines.
- **Effectiveness under Calibration**: Once the logits are calibrated, confidence and entropy represent excellent proxies for failure detection.

### 2.1.3 Weaknesses
- **Blindness to Model Disagreement**: Single-pass models cannot distinguish whether a prediction is uncertain due to data noise (aleatoric) or model ignorance (epistemic).
- **Overconfidence on Out-of-Distribution (OOD) Samples**: Standard neural networks map features to high-dimensional space where far-away OOD samples can still fall into high-activation zones, producing highly confident but incorrect predictions.

---

## 2.2 Stochastic Uncertainty Methods (Monte Carlo Dropout)

Stochastic uncertainty treats the network weights as random variables. Following Gal & Ghahramani (2016), applying dropout during training and inference acts as a variational approximation to a deep Gaussian Process.

### 2.2.1 Core Paradigm
- **Training**: Standard training with dropout layers active.
- **Inference**: Instead of disabling dropout (setting `model.eval()`), we selectively keep only the dropout layers active (`dropout.train()`) while keeping BatchNorm and other parameters in evaluation mode (`model.eval()`).
- **Sampling**: We execute $N$ forward passes ($W^{(t)} \sim q(W)$), generating $N$ different logit vectors $z_1 \dots z_N$.
- **Calibration Integration**: Each logit vector is calibrated by temperature scaling before the softmax is applied:
  $$p_{t} = \text{softmax}(z_t / T)$$
- **Ensembling**: We average the probabilities:
  $$\bar{p} = \frac{1}{N} \sum_{t=1}^N p_t$$

### 2.2.2 Strengths
- **Isolates Epistemic Uncertainty**: Through Mutual Information (MI), MC Dropout quantifies how much the model's parameters disagree, indicating whether the sample lies in a poorly explored region of the feature space.
- **Improved OOD Robustness**: In regions lacking training data, stochastic weights produce highly variable predictions, lowering calibrated confidence and increasing Mutual Information.
- **Empirical Failure Detection**: By aggregating multiple predictions, it captures borderline inputs (e.g., cases lying on the boundary between Mild and Moderate Diabetic Retinopathy) more robustly.

### 2.2.3 Weaknesses
- **High Computational Overhead**: Requires $N$ forward passes ($O(N)$ complexity), increasing latency and compute cost by a factor of $N$. For clinical edge deployment (e.g., screening cameras), this can be restrictive.
- **Dependency on Dropout Configuration**: If dropout layers are missing, or placed in suboptimal positions, or if the dropout rate is too low, the passes will not produce meaningful variance.

---

## 2.3 Methodological Comparison Matrix

| Property | Deterministic Baseline | MC Dropout ($N=25$) |
| :--- | :---: | :---: |
| **Inference Cost** | $1\times$ (Base latency) | $25\times$ Latency |
| **Data Requirements** | None (Post-hoc calibration only) | Requires dropout during training |
| **Uncertainty Splits** | None (Combines all sources) | Decomposes Aleatoric vs. Epistemic |
| **T-Scaling Application** | Once on output logits | Inside every pass before softmax |
| **OOD Behavior** | Susceptible to overconfidence | Lower confidence, high variance |
| **Borderline Case Sensitivity** | Captured via top-2 margin | Captured via stochastic disagreement |
