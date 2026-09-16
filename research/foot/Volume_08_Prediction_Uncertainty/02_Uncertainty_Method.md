# Chapter 02 — Uncertainty Formulations & Method

## 1. Option B Calibrated MC Dropout Pipeline

In post-hoc uncertainty estimation for calibrated classifiers, two integration options exist:

- **Option A (Uncalibrated MC -> Softmax -> Average -> Post-hoc Calibrate)**: Calibrates the averaged probability distribution post-hoc.
- **Option B (Stochastic MC Logits -> Frozen Calibrator -> Softmax -> Average & Metrics)**: Applies frozen Vector Scaling ($w^{\ast}, b^{\ast}$) to every individual stochastic forward logit pass $z_t$ before softmax normalization:

$$ z'_{t, k} = w_k^{\ast} z_{t, k} + b_k^{\ast}, \quad p_{t, k} = \frac{\exp(z'_{t, k})}{\sum_{j=1}^4 \exp(z'_{t, j})} $$

Option B is selected as the canonical pipeline because it ensures every single stochastic pass yields a valid, calibrated probability distribution prior to uncertainty decomposition.

---

## 2. Mathematical Metrics & Formulations

For $N$ stochastic forward passes on sample $i$:

### 2.1 Predictive Mean Probability ($\bar{p}_i$)
$$ \bar{p}_{i, k} = \frac{1}{N} \sum_{t=1}^N p_{i, t, k} $$

### 2.2 Predictive Variance ($\text{Var}(p_i)$)
Average class probability variance across $N$ passes:
$$ \text{Var}(p_i) = \frac{1}{K} \sum_{k=1}^K \left[ \frac{1}{N - 1} \sum_{t=1}^N (p_{i, t, k} - \bar{p}_{i, k})^2 \right] $$

### 2.3 Total Predictive Entropy ($H(\bar{p}_i)$)
Entropy of the predictive mean (captures total uncertainty):
$$ H(\bar{p}_i) = - \sum_{k=1}^K \bar{p}_{i, k} \log \bar{p}_{i, k} $$

### 2.4 Aleatoric Uncertainty ($\mathbb{E}[H(p_{i, t})]$)
Expected entropy across individual stochastic passes (data / observation noise):
$$ \mathbb{E}[H(p_{i, t})] = \frac{1}{N} \sum_{t=1}^N \left( - \sum_{k=1}^K p_{i, t, k} \log p_{i, t, k} \right) $$

### 2.5 Epistemic Uncertainty / Mutual Information ($MI_i$)
Information gain between model parameters and predictive distribution (model / parameter uncertainty):
$$ MI_i = H(\bar{p}_i) - \mathbb{E}[H(p_{i, t})] $$

All metrics satisfy non-negativity bounds: $\text{Var}(p_i) \ge 0$, $0 \le H(\bar{p}_i) \le \log(4) \approx 1.3863$, $MI_i \ge 0$.
