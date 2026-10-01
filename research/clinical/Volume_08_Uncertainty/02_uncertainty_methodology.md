# Document 02: Uncertainty Methodology & Mathematical Formulation

## 1. Ensemble Resampling Formulation

To quantify model parameter-estimation uncertainty on tabular gradient-boosted decision trees, we employ a non-parametric **Bootstrap Ensemble**.

Given the training dataset $\mathcal{D}_{\text{train}} = \{(x_i, y_i)\}_{i=1}^{N_{\text{train}}}$ with $N_{\text{train}} = 69,519$, we generate $M = 50$ bootstrap datasets $\mathcal{D}_{\text{train}}^{(1)}, \dots, \mathcal{D}_{\text{train}}^{(M)}$ by uniform sampling with replacement:

$$\mathcal{D}_{\text{train}}^{(m)} \sim \text{Bootstrap}(\mathcal{D}_{\text{train}}), \quad |\mathcal{D}_{\text{train}}^{(m)}| = N_{\text{train}}$$

For each resample $m \in \{1, \dots, M\}$, an independent `CatBoostClassifier` model $f_m(x)$ is fitted with frozen hyperparameters and distinct random seed $S_m = 42 + m$.

---

## 2. Stochastic Prediction Statistics

For any patient encounter with feature vector $x \in \mathbb{R}^{119}$, the ensemble generates a vector of probability predictions:

$$\mathbf{p}(x) = \left[ p_1(x), p_2(x), \dots, p_M(x) \right]^T \in [0, 1]^M$$

### 2.1 Ensemble Mean Prediction
$$\bar{p}(x) = \frac{1}{M} \sum_{m=1}^M p_m(x)$$

### 2.2 Predictive Variance & Standard Deviation
$$\text{Var}[p(x)] = \frac{1}{M-1} \sum_{m=1}^M \left( p_m(x) - \bar{p}(x) \right)^2$$
$$\sigma_p(x) = \sqrt{\text{Var}[p(x)]}$$

Here, $\sigma_p(x)$ (the standard deviation of predicted probability across bootstrap models) serves as the primary metric of **bootstrap ensemble uncertainty**.

### 2.3 Bootstrap Predictive Interval (95% Envelope)
$$I_{95}(x) = \left[ \text{Quantile}_{0.025}(\mathbf{p}(x)), \; \text{Quantile}_{0.975}(\mathbf{p}(x)) \right]$$

This interval captures empirical prediction dispersion across plausible model parameter configurations.

### 2.4 Aleatoric Entropy Proxy
To assess intrinsic outcome ambiguity associated with the predicted probability level:
$$\mathcal{H}(\bar{p}(x)) = -\bar{p}(x) \log_2 \bar{p}(x) - (1 - \bar{p}(x)) \log_2 (1 - \bar{p}(x))$$

---

## 3. Calibration-Aware Uncertainty Integration

To preserve consistency with Phase C7:
- **Option A (Frozen Calibration Mapping)**: A single calibration function $g_{\text{cal}}: [0, 1] \to [0, 1]$ (**Isotonic Regression**, the validation NLL winner) is fitted on validation mean predictions $(\bar{p}_{\text{val}}, y_{\text{val}})$ and applied to transform ensemble mean probabilities:
  $$\hat{p}_{\text{cal}}(x) = g_{\text{cal}}(\bar{p}(x))$$
- Uncertainty statistics ($\sigma_p(x)$, $I_{95}(x)$) quantify the underlying ensemble dispersion.

---

## 4. Uncertainty Evaluation Metrics

### 4.1 Error Detection AUROC / AUPRC
Let binary misclassification at operating threshold $\theta$ be defined as $e_i = \mathbb{I}(\hat{y}_i \neq y_i)$, where $\hat{y}_i = \mathbb{I}(\bar{p}_i \ge \theta)$.
- **Error Detection AUROC**: Area under the ROC curve treating $e_i$ as target and $\sigma_p(x_i)$ as predictor score:
  $$\text{AUROC}_{\text{error}} = \mathbb{P}\left(\sigma_p(x_{\text{incorrect}}) > \sigma_p(x_{\text{correct}})\right)$$
- **Error Detection AUPRC**: Area under the Precision-Recall curve for detecting classification errors.

### 4.2 Risk-Coverage Analysis & AURC
Encounters are sorted by uncertainty ascending: $\sigma_p(x_{(1)}) \le \sigma_p(x_{(2)}) \le \dots \le \sigma_p(x_{(N)})$. For coverage $c \in (0, 1]$, the selective classifier retains the top $\lfloor c \cdot N \rfloor$ most confident predictions:

$$\text{Risk}(c) = \frac{1}{\lfloor c \cdot N \rfloor} \sum_{i=1}^{\lfloor c \cdot N \rfloor} \mathbb{I}\left( \hat{y}_{(i)} \neq y_{(i)} \right)$$

$$\text{AURC} = \int_{0}^1 \text{Risk}(c) \, dc$$
$$\text{E-AURC} = \text{AURC} - \text{AURC}_{\text{optimal}}$$
