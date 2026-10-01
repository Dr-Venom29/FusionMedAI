# Document 02: Calibration Methodology & Metrics

## 1. Overview of Calibration Methods

To convert raw decision tree margins into statistically reliable probabilities, three canonical calibration algorithms are implemented in `src/clinical/calibration/calibrator.py`.

---

## 2. Mathematical Formulations

### 2.1 Platt Scaling (Logistic Calibration)
Platt scaling models the calibrated probability as a univariate logistic transformation of the raw model logit $z_i = \text{logit}(p_i) = \ln(p_i / (1 - p_i))$:

$$g_{\text{Platt}}(z_i) = \sigma(a \cdot z_i + b) = \frac{1}{1 + \exp(-(a \cdot z_i + b))}$$

where $a \in \mathbb{R}^+$ and $b \in \mathbb{R}$ are estimated by minimizing the cross-entropy loss over the validation partition:

$$\mathcal{L}_{\text{Platt}}(a, b) = -\sum_{i=1}^{N_{\text{val}}} \left[ y_i \ln \sigma(a z_i + b) + (1 - y_i) \ln(1 - \sigma(a z_i + b)) \right]$$

- **Properties**: Strictly monotonic when $a > 0$; preserves rank ordering and smooth probability density.
- **Limitation**: Assumes a sigmoid distortion profile; cannot correct non-sigmoidal S-curves.

---

### 2.2 Isotonic Regression (PAVA)
Isotonic regression is a non-parametric approach that fits a piecewise constant, monotonically non-decreasing mapping $g_{\text{Iso}}: [0, 1] \to [0, 1]$ using the **Pool Adjacent Violators Algorithm (PAVA)**:

$$\min_{g \in \mathcal{M}} \sum_{i=1}^{N_{\text{val}}} (y_i - g(p_i))^2 \quad \text{subject to } g(p_i) \le g(p_j) \text{ whenever } p_i \le p_j$$

where $\mathcal{M}$ is the class of all isotonic (monotonically non-decreasing) step functions.

- **Properties**: Completely non-parametric; can correct arbitrary monotonic distortions without parametric assumptions.
- **Limitation**: Produces discrete step functions; risks overfitting small validation bins and may introduce ties among previously distinct probability ranks.

---

### 2.3 Beta Calibration
Beta calibration is designed specifically for binary classifiers where score distributions approximate Beta distributions under positive and negative classes. The calibrated probability is given by:

$$g_{\text{Beta}}(p_i) = \frac{1}{1 + \frac{1}{\exp(c)} \cdot \frac{(1 - p_i)^b}{p_i^a}} = \sigma(a \ln p_i - b \ln(1 - p_i) + c)$$

where $a, b \ge 0$ and $c \in \mathbb{R}$ are fitted via maximum likelihood on $(p_{\text{val}}, y_{\text{val}})$.

- **Properties**: Generalizes Platt scaling (when $a = b$); handles skewed distributions and asymmetric boundary compression.
- **Limitation**: Requires optimization of 3 non-linear parameters with boundary constraints.

---

## 3. Statistical Reliability & Calibration Metrics

### 3.1 Brier Score
The mean squared error of predicted probabilities:
$$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2$$
The Brier score decomposes into $\text{Reliability} - \text{Resolution} + \text{Uncertainty}$. Lower values indicate superior combined discrimination and calibration.

### 3.2 Negative Log-Likelihood (Log Loss)
The standard cross-entropy metric:
$$\text{Log Loss} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln \hat{p}_i + (1 - y_i) \ln(1 - \hat{p}_i) \right]$$
Strictly proper scoring rule sensitive to overconfident misclassifications.

### 3.3 Expected Calibration Error (ECE)
Samples are partitioned into $M=10$ equally spaced bins $B_m = (\frac{m-1}{M}, \frac{m}{M}]$:
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
where $\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} y_i$ is the empirical event rate and $\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i$ is the mean predicted probability in bin $m$.

### 3.4 Maximum Calibration Error (MCE)
The worst-case bin deviation:
$$\text{MCE} = \max_{m \in \{1, \dots, M\}} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

### 3.5 Logistic Calibration Intercept & Slope
Regressing true labels against predicted log-odds:
$$\text{logit}(y_i) \sim \alpha + \beta \cdot \text{logit}(\hat{p}_i)$$
- **Calibration Intercept ($\alpha$)**: Measures overall calibration-in-the-large. Ideal value is $\alpha = 0$. $\alpha < 0$ indicates systematic overestimation of risk.
- **Calibration Slope ($\beta$)**: Measures spread of probabilities. Ideal value is $\beta = 1.0$. $\beta < 1.0$ indicates overly extreme predictions (too low for low-risk, too high for high-risk); $\beta > 1.0$ indicates under-confident predictions.
