# Document 03: Uncalibrated Baseline Analysis

## 1. Raw CatBoost Risk Calibration Profile

The frozen candidate model—CatBoost with shallow symmetric oblivious trees (`depth=4`, `iterations=350`, `learning_rate=0.1383`)—was trained with binary cross-entropy loss (LogLoss).

Gradient boosted trees with shallow depth and strong $L_2$ leaf regularization (`l2_leaf_reg=2.911`) often exhibit strong natural calibration compared to deep neural networks, because shallow tree splits conservatively approximate local log-odds without extreme logit saturation.

---

## 2. Empirical Baseline Metrics

The table below summarizes the uncalibrated baseline performance on the validation ($N=14,911$) and locked test ($N=14,913$) partitions:

| Metric | Validation Set ($N=14,911$) | Locked Test Set ($N=14,913$) | Optimal Value |
| :--- | :---: | :---: | :---: |
| **Log Loss (NLL)** | 0.343665 | 0.333780 | $0.0$ |
| **Brier Score** | 0.099026 | 0.095340 | $0.0$ |
| **Expected Calibration Error (ECE)** | 0.004802 | 0.003198 | $0.0$ |
| **Maximum Calibration Error (MCE)** | 0.878221 | 0.334020 | $0.0$ |
| **Calibration Intercept ($\alpha$)** | -0.002721 | -0.125387 | $0.0$ |
| **Calibration Slope ($\beta$)** | 0.983834 | 0.949153 | $1.0$ |
| **ROC-AUC** | 0.650231 | 0.649503 | $1.0$ |
| **PR-AUC** | 0.217551 | 0.203516 | $1.0$ |

---

## 3. Reliability Analysis & Diagnostic Observations

### 3.1 Slope & Spread
On the validation set, the raw CatBoost model achieves a calibration slope of $\beta = 0.9838 \approx 1.0$, indicating that the log-odds spread aligns closely with observed cohort frequencies. On the test set, the slope shifts slightly to $\beta = 0.9492$, reflecting minor over-dispersion in the tail regions.

### 3.2 Calibration-in-the-Large (Intercept)
The validation intercept is $\alpha = -0.0027$, indicating near-zero global bias. On the test set, the intercept is $\alpha = -0.1254$, driven by a minor shift in test cohort prevalence ($11.11\%$ test vs $11.55\%$ validation).

### 3.3 Reliability Curve Diagnostics
As visualized in the uncalibrated panel of the 4-panel reliability diagram (`figures/reliability_diagrams.png`), the raw probability curve tracks the 45-degree diagonal reference line across the primary density interval $[0.05, 0.30]$. 

```
Predicted Risk [0.00 - 0.10]: Empirical Rate ≈  7.5% (Well-aligned)
Predicted Risk [0.10 - 0.20]: Empirical Rate ≈ 14.8% (Well-aligned)
Predicted Risk [0.20 - 0.30]: Empirical Rate ≈ 24.1% (Slight underestimation)
Predicted Risk [0.30 - 0.50]: Empirical Rate ≈ 38.2% (Moderate variance due to sample scarcity)
```

### 3.4 Summary
The raw CatBoost HPO model provides a competitive baseline with $\text{ECE} = 0.0032$ and $\text{Brier} = 0.0953$, establishing a high benchmark for post-hoc calibration methods.
