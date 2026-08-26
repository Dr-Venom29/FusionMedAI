# 4. Uncertainty Metrics: Formulation and Code Implementation

This chapter documents the mathematical definitions, code implementations, and operational orientations of the uncertainty metrics analyzed in Step 8.

---

## 4.1 Shannon Entropy (Predictive Uncertainty)

Shannon entropy captures the aggregate uncertainty of a probability distribution.

### 4.1.1 Formulation
For a probability vector $p = [p_1, \dots, p_K]$ where $K = 5$ is the number of Diabetic Retinopathy classes:

$$H(p) = -\sum_{c=1}^K p_c \log_e(p_c)$$

To make the metric bounded and scale-invariant, we divide by the maximum possible entropy:

$$H_{\text{norm}}(p) = \frac{H(p)}{\log_e(K)}$$

where $H_{\text{norm}}(p) \in [0, 1]$.

### 4.1.2 Python Implementation
```python
def compute_entropy_np(probs: np.ndarray) -> np.ndarray:
    eps = 1e-9
    clipped_probs = np.clip(probs, eps, 1.0 - eps)
    entropy = -np.sum(clipped_probs * np.log(clipped_probs), axis=-1)
    return entropy
```

---

## 4.2 Monte Carlo Decompositions

For $N$ stochastic predictions $p_1, \dots, p_N$ with mean probability $\bar{p} = \frac{1}{N}\sum_t p_t$:

### 4.2.1 Predictive Entropy
Total uncertainty of the ensemble mean prediction:

$$H(\bar{p}) = -\sum_{c=1}^K \bar{p}_c \log_e(\bar{p}_c)$$

### 4.2.2 Expected Entropy
The average aleatoric (data noise) uncertainty across stochastic configurations:

$$\mathbb{E}[H(p)] = \frac{1}{N}\sum_{t=1}^N H(p_t)$$

### 4.2.3 Mutual Information (Epistemic Uncertainty)
Isolates model knowledge gaps (disagreement between passes):

$$MI = H(\bar{p}) - \mathbb{E}[H(p)]$$

### 4.2.4 Predictive Variance (Mean Probability Variance)
Measures the class-wise variance across passes, averaged across classes:

$$\text{Var}_{\text{mean}}(P) = \frac{1}{K}\sum_{c=1}^K \left[ \frac{1}{N}\sum_{t=1}^N (p_{t,c} - \bar{p}_c)^2 \right]$$

### 4.2.5 Python Implementation
```python
# mean_probs shape: (num_samples, num_classes)
mean_probs = np.mean(all_mc_probs, axis=1)
predictive_entropy = compute_entropy_np(mean_probs)

# Compute individual pass entropies
pass_entropies = np.zeros((num_samples, n_passes))
for t in range(n_passes):
    pass_entropies[:, t] = compute_entropy_np(all_mc_probs[:, t, :])
expected_entropy = np.mean(pass_entropies, axis=1)

# Epistemic Uncertainty
mutual_information = np.clip(predictive_entropy - expected_entropy, 0.0, None)

# Predictive Variance
class_variances = np.var(all_mc_probs, axis=1) # (num_samples, num_classes)
predictive_variance = np.mean(class_variances, axis=1) # (num_samples,)
```

---

## 4.3 Score Orientations for Error Detection

To evaluate uncertainty as a binary classifier for prediction failure (where the positive class is `is_incorrect == 1`), all metrics must be oriented consistently so that a **higher score indicates higher uncertainty (greater likelihood of error)**.

We define the mappings and orientations below:

| Metric | Raw Variable | Orientation | Score Used for ROC ($s$) |
| :--- | :---: | :---: | :---: |
| **Calibrated Confidence** | $C$ | Lower = More Uncertain | $1.0 - C$ |
| **Top-2 Margin** | $M$ | Lower = More Uncertain | $1.0 - M$ |
| **Shannon Entropy** | $H_{\text{norm}}$ | Higher = More Uncertain | $H_{\text{norm}}$ |
| **Predictive Entropy** | $H(\bar{p})$ | Higher = More Uncertain | $H(\bar{p})$ |
| **Expected Entropy** | $\mathbb{E}[H(p)]$ | Higher = More Uncertain | $\mathbb{E}[H(p)]$ |
| **Predictive Variance** | $\text{Var}_{\text{mean}}$ | Higher = More Uncertain | $\text{Var}_{\text{mean}}$ |
| **Mutual Information** | $MI$ | Higher = More Uncertain | $MI$ |

---

## 4.4 Risk-Coverage and Area Under Curve (AURC)

Selective prediction allows the model to abstain from predicting if the uncertainty exceeds a threshold. We define:

1. **Uncertainty Ranking**: Sort all test samples in ascending order of their uncertainty score $s$ (most certain first).
2. **Coverage ($c$)**: The fraction of samples retained:
   $$c = \frac{k}{M}$$
   where $k$ is the number of retained samples, and $M$ is the test split size.
3. **Risk ($\text{Risk}(c)$)**: The error rate of the retained samples:
   $$\text{Risk}(c) = \frac{1}{k}\sum_{j=1}^k \mathbb{I}(\hat{y}_j \neq y_j)$$
4. **Area Under Risk-Coverage Curve (AURC)**:
   Integrates the risk across all coverage levels:
   $$\text{AURC} = \frac{1}{M}\sum_{i=1}^M \text{Risk}\left(\frac{i}{M}\right)$$
5. **Excess AURC (E-AURC)**:
   Quantifies the gap between the evaluated method and an **optimal ranker** (which perfectly knows which predictions are incorrect and rejects them first):
   $$\text{E-AURC} = \text{AURC} - \text{AURC}_{\text{opt}}$$
   where $\text{AURC}_{\text{opt}}$ is the AURC of the optimal ranker. E-AURC values closer to 0 represent superior selective prediction capability.
