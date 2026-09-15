# Chapter 02 — Calibration Method Formulations

## 2.1 Baseline: Method A — Uncalibrated Raw Softmax

Let $z_i \in \mathbb{R}^K$ be the vector of unnormalized class logits produced by EfficientNet-B3 for input sample $x_i$, where $K = 4$ Wagner classes.

The raw predicted class probabilities $p_i$ are computed via standard softmax:

$$p_i = \text{softmax}(z_i) = \frac{\exp(z_{i,c})}{\sum_{k=1}^K \exp(z_{i,k})}$$

---

## 2.2 Method B — Temperature Scaling

Temperature Scaling (Guo et al., 2017) applies a single positive scalar parameter $T > 0$ to scale all unnormalized logits uniformly before the softmax transformation:

$$\hat{p}_i(T) = \text{softmax}\left( \frac{z_i}{T} \right)$$

### Parameter Optimization
The optimal temperature $T^*$ is optimized on the validation set by minimizing Cross-Entropy (Negative Log-Likelihood):

$$T^* = \arg\min_T -\frac{1}{N_{\text{val}}} \sum_{i=1}^{N_{\text{val}}} \log \left( \text{softmax}\left( \frac{z_i}{T} \right)_{y_i} \right)$$

- $T > 1$: Softens probability distribution (reduces overconfidence).
- $T < 1$: Sharpens probability distribution (increases confidence).
- $T = 1$: Equivalent to raw softmax baseline.

To strictly enforce $T > 0$, optimization is performed over $\log T \in \mathbb{R}$ using L-BFGS.

---

## 2.3 Method C — Vector Scaling

Vector Scaling extends temperature scaling by introducing a diagonal weight matrix $W = \text{diag}(w_1, \dots, w_K) \in \mathbb{R}^{K \times K}$ and a class-specific bias vector $b \in \mathbb{R}^K$:

$$z'_i = W z_i + b = w \odot z_i + b$$
$$\hat{p}_i(W, b) = \text{softmax}(z'_i)$$

### Parameter Optimization
The parameters $w^* \in \mathbb{R}^4$ and $b^* \in \mathbb{R}^4$ (8 learnable parameters total) are optimized on validation logits via L-BFGS minimizing NLL:

$$\{w^*, b^*\} = \arg\min_{w, b} -\frac{1}{N_{\text{val}}} \sum_{i=1}^{N_{\text{val}}} \log \left( \text{softmax}(w \odot z_i + b)_{y_i} \right)$$

Vector scaling allows class-specific confidence adjustments, providing greater flexibility than a single temperature scalar.
