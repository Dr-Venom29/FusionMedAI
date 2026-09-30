# Chapter 03 — End-to-End Inference Execution Pipeline

## 1. Execution Flow Overview

The inference execution inside `FootModule.predict()` follows a 4-stage sequential pipeline:

```
                  ┌──────────────────────────────┐
                  │ 1. Input Image Validation &   │
                  │    224x224 RGB Normalization │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │ 2. Deterministic Inference   │
                  │    & Vector Scaling          │
                  │    - Raw Logits z            │
                  │    - Scaled Logits z'        │
                  │    - Calibrated Probs p      │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │ 3. Stochastic MC Dropout     │
                  │    Uncertainty (N*=10)       │
                  │    - Dropout -> train()      │
                  │    - BatchNorm -> eval()     │
                  │    - Vector Scaled Pass Logits│
                  │    - H(p), Var(p), MI        │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │ 4. Deterministic Grad-CAM    │
                  │    Explainability Overlay    │
                  │    - Hook target layer 8     │
                  │    - JET Color Blending      │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │ 5. Unified Schema Output     │
                  └──────────────────────────────┘
```

---

## 2. Stage Details

### Stage 1 — Image Preprocessing

Input images are loaded, converted to RGB, resized to $224 \times 224$, and normalized using measured dataset statistics:

$$
\mu = [0.4937, 0.3630, 0.3272], \quad \sigma = [0.1745, 0.1632, 0.1551]
$$

### Stage 2 — Deterministic Inference & Vector Scaling

Under `model.eval()`, the forward pass computes raw logits $z \in \mathbb{R}^4$. Calibrated logits $z'$ and probabilities $p_{\text{calib}}$ are produced using the frozen Vector Scaling parameters:

$$
z'_k = w_k^{\ast} \cdot z_k + b_k^{\ast}
$$

$$
p_{\text{calib}, k} = \frac{\exp(z'_k)}{\sum_{j=1}^4 \exp(z'_j)}
$$

### Stage 3 — Stochastic MC Dropout Uncertainty Estimation

When $N^{\ast} = 10$, `enable_foot_mc_dropout(model)` sets `nn.Dropout` modules to `.train()` while preserving `BatchNorm` in `.eval()`. For each stochastic pass $t \in \{1 \dots N^{\ast}\}$, Vector Scaling is applied to pass logits $z_t$:

$$
z'_{t, k} = w_k^{\ast} \cdot z_{t, k} + b_k^{\ast}, \quad p_{t, k} = \text{softmax}(z'_t)_k
$$

Across $N^{\ast} = 10$ passes, metrics are computed:

- **Predictive Mean ($\bar{p}$)**:

  $$
  \bar{p} = \frac{1}{N^{\ast}} \sum_{t=1}^{N^{\ast}} p_t
  $$

- **Total Predictive Entropy ($H(\bar{p})$)**:

  $$
  H(\bar{p}) = -\sum_{k=1}^4 \bar{p}_k \log \bar{p}_k
  $$

- **Predictive Variance ($\text{Var}(p)$)**:

  $$
  \text{Var}(p) = \frac{1}{4} \sum_{k=1}^4 \text{Var}_t(p_{t, k})
  $$

- **Epistemic Mutual Information ($MI$)**:

  $$
  MI = H(\bar{p}) - \frac{1}{N^{\ast}} \sum_{t=1}^{N^{\ast}} H(p_t)
  $$

### Stage 4 — Grad-CAM Explainability Overlay

If `generate_cam=True`, `FootGradCAM` extracts spatial activation maps from `backbone.features.8` for predicted class $\hat{Y} = \arg\max(p_{\text{calib}})$. The normalized CAM map $M \in [0, 1]^{224 \times 224}$ is colored via OpenCV `COLORMAP_JET` and blended with original image $I_{\text{orig}}$:

$$
I_{\text{overlay}} = \text{uint8}(0.6 \cdot I_{\text{orig}} + 0.4 \cdot H_{\text{JET}})
$$
