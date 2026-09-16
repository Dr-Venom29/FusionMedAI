# Chapter 02 — Calibration Method Formulations

## 2.1 Baseline: Method A — Uncalibrated Raw Softmax

Let $z_i \in \mathbb{R}^K$ be the vector of unnormalized class logits produced by EfficientNet-B3 for input sample $x_i$, where $K = 4$ Wagner classes.

The raw predicted probability for class $c$ is computed via standard softmax:

$$p_{i,c} = \frac{\exp(z_{i,c})}{\sum_{k=1}^K \exp(z_{i,k})}$$

---

## 2.2 Method B — Temperature Scaling

Temperature Scaling (Guo et al., 2017) applies a single positive scalar parameter $T > 0$ to scale all unnormalized logits uniformly before the softmax transformation:

$$\hat{p}_{i,c}(T) = \frac{\exp(z_{i,c} / T)}{\sum_{k=1}^K \exp(z_{i,k} / T)}$$

### Parameter Optimization
The optimal temperature $T^*$ is optimized on the validation set by minimizing Negative Log-Likelihood (NLL):

$$T^* = \arg\min_T -\frac{1}{N_{\text{val}}} \sum_{i=1}^{N_{\text{val}}} \log \hat{p}_{i, y_i}(T)$$

- $T > 1$: Softens probability distribution (reduces overconfidence).
- $T < 1$: Sharpens probability distribution (increases confidence).
- $T = 1$: Equivalent to raw softmax baseline.

To strictly enforce $T > 0$, optimization is performed over $\log T \in \mathbb{R}$ using L-BFGS.

---

## 2.3 Method C — Vector Scaling

Vector Scaling extends temperature scaling by applying a per-class weight vector $w \in \mathbb{R}^K$ and bias vector $b \in \mathbb{R}^K$:

$$z'_{i,c} = w_c \cdot z_{i,c} + b_c$$

$$\hat{p}_{i,c}(w, b) = \frac{\exp(z'_{i,c})}{\sum_{k=1}^K \exp(z'_{i,k})}$$

### Parameter Optimization
The parameters $w^* \in \mathbb{R}^4$ and $b^* \in \mathbb{R}^4$ (8 learnable parameters total) are optimized on validation logits via L-BFGS minimizing NLL:

$$(w^*, b^*) = \arg\min_{w, b} -\frac{1}{N_{\text{val}}} \sum_{i=1}^{N_{\text{val}}} \log \hat{p}_{i, y_i}(w, b)$$

Vector scaling allows class-specific confidence adjustments, providing greater flexibility than a single temperature scalar.
