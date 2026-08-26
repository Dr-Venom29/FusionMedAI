# 1. Theoretical Foundations of Uncertainty Estimation

In deep learning for medical diagnostics, understanding *when* a model is likely to fail is as critical as its overall accuracy. Prediction uncertainty estimation provides a mathematical framework for quantifying the confidence of neural network predictions. This volume divides uncertainty signals into **deterministic predictive uncertainty proxies** and **stochastic uncertainty metrics** computed via Monte Carlo (MC) Dropout.

## 1.1 Types of Uncertainty

In Bayesian deep learning, prediction uncertainty is typically categorized into two distinct forms:

1. **Aleatoric Uncertainty (Data Noise)**:
   - Represents the intrinsic randomness or noise in the data (e.g., sensor noise, poor image quality, motion artifacts in retinal scans, or ambiguous clinical features).
   - Cannot be reduced by collecting more training data of the same quality.
   - Mathematically captured by high entropy in the predictive distribution, or high expected entropy across stochastic samples.

2. **Epistemic Uncertainty (Model Knowledge)**:
   - Represents the model's lack of knowledge about the data-generating process (e.g., out-of-distribution samples, classes not well-represented in the training set).
   - Can be reduced by collecting more training data in that region of the feature space.
   - Mathematically captured by the variance of predictions across stochastic passes (model disagreement), or by Mutual Information (MI).

---

## 1.2 Deterministic Uncertainty Proxies

For a standard neural network producing class probabilities $P(y|x)$ (possibly scaled by a post-hoc calibration temperature $T$), we define three deterministic proxies:

### 1.2.1 Calibrated Confidence
Calibrated confidence is defined as the maximum probability of the calibrated predictive distribution:

$$C(x) = \max_{c \in \{1,\dots,K\}} P(y=c \mid x, T)$$

where $K = 5$ is the number of Diabetic Retinopathy classes. Lower confidence indicates higher uncertainty.

### 1.2.2 Shannon Entropy
Shannon entropy measures the overall information content or "flatness" of the probability distribution:

$$H(P) = -\sum_{c=1}^K p_c \log_e(p_c)$$

To make comparisons across architectures scale-invariant, we normalize entropy by the maximum possible entropy (which occurs under a uniform distribution $H_{\text{max}} = \log_e(K)$):

$$H_{\text{norm}}(P) = \frac{H(P)}{\log_e(K)}$$

where $H_{\text{norm}}(P) \in [0, 1]$. Higher entropy indicates greater model hesitation.

### 1.2.3 Top-2 Margin
The top-2 margin is the difference between the probabilities of the most likely class and the second most likely class:

$$\text{Margin}(P) = p_{\text{top1}} - p_{\text{top2}}$$

A small margin indicates that the prediction is borderline between two classes (grade confusion).

---

## 1.3 Stochastic Uncertainty Metrics (MC Dropout)

Stochastic uncertainty estimation models the network weights as random variables $W$ drawn from a posterior distribution $q(W)$. During inference, we draw $N$ stochastic weight configurations $W^{(t)} \sim q(W)$ by enabling dropout layers.

For an input retinal scan $x$, each pass $t$ yields a calibrated class probability vector:

$$p_t = \text{softmax}\left(\frac{f(x; W^{(t)})}{T}\right)$$

The overall **predictive distribution** is the mean of these stochastic passes:

$$\bar{p} = \frac{1}{N} \sum_{t=1}^N p_t$$

We compute the following stochastic uncertainty metrics:

### 1.3.1 Predictive Entropy
Predictive entropy measures the total uncertainty in the average stochastic prediction:

$$H(\bar{p}) = -\sum_{c=1}^K \bar{p}_c \log_e(\bar{p}_c)$$

This combines both aleatoric and epistemic uncertainty.

### 1.3.2 Expected Entropy
Expected entropy measures the average uncertainty within individual stochastic predictions:

$$\mathbb{E}_{q(W)}[H(p)] \approx \frac{1}{N} \sum_{t=1}^N H(p_t)$$

where $H(p_t) = -\sum_{c=1}^K p_{t,c} \log_e(p_{t,c})$. Expected entropy represents aleatoric uncertainty (data ambiguity).

### 1.3.3 Mutual Information (Epistemic Uncertainty)
Mutual Information (MI) measures the disagreement between stochastic passes, isolating epistemic uncertainty:

$$MI(y, W \mid x) = H(\bar{p}) - \mathbb{E}_{q(W)}[H(p)]$$

If the stochastic predictions are highly confident but disagree on *which* class is correct (e.g., some passes predict Grade 1 and others predict Grade 3), $H(\bar{p})$ will be high but the expected entropy $\mathbb{E}[H(p)]$ will be low. This results in high Mutual Information, signaling model ignorance.

### 1.3.4 Predictive Variance (Mean Probability Variance)
Predictive variance measures the average variance of class probabilities across the $N$ passes:

$$\text{Var}_{\text{mean}}(P) = \frac{1}{K} \sum_{c=1}^K \text{Var}(p_{\cdot, c})$$

where:

$$\text{Var}(p_{\cdot, c}) = \frac{1}{N} \sum_{t=1}^N \left(p_{t,c} - \bar{p}_c\right)^2$$

This provides a direct measure of prediction dispersion.
