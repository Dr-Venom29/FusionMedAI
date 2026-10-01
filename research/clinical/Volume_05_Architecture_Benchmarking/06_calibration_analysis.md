# Research Document 06: Probability Calibration & Reliability Analysis

## 1. Clinical Importance of Probability Calibration

In clinical decision support systems, discrimination (ROC-AUC) is insufficient on its own. Clinicians, triage nurses, and care-management algorithms require **calibrated risk probabilities**:
- If an algorithm outputs a predicted readmission probability of $\hat{p} = 0.25$ for $100$ hospitalized diabetic patients, exactly $\approx 25$ of those patients should realistically be readmitted within $30$ days.
- Overconfident models cause **alarm fatigue** and wasteful resource allocation; underconfident models cause clinicians to miss high-risk deteriorating patients.

---

## 2. Calibration Metrics & Mathematical Definitions

### A. Expected Calibration Error (ECE)
The probability space $[0, 1]$ is partitioned into $M=10$ equal-width or quantile bins $B_1, B_2, \dots, B_M$. ECE computes the sample-weighted absolute difference between empirical accuracy and mean predicted confidence:
$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
Where:
$$\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} y_i, \quad \text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i$$

### B. Brier Score
The mean squared difference between predicted probabilities and binary outcomes ($y_i \in \{0, 1\}$):
$$\text{BS} = \frac{1}{N} \sum_{i=1}^{N} (\hat{p}_i - y_i)^2$$
A lower Brier score denotes superior joint calibration and refinement.

### C. Logarithmic Loss (Cross-Entropy)
Penalizes severe overconfidence on incorrect predictions:
$$\text{LogLoss} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

---

## 3. Calibration Scoreboard Across 7 Architectures

Evaluated on the locked test partition ($N=14,913$, $1,664$ positive readmissions):

| Architecture | Model Family | Test Brier Score | Test Log-Loss | Test ECE (10 Bins) | Calibration Rank |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | GBDT (Histogram) | $\mathbf{0.0953}$ | $0.3339$ | $\mathbf{0.0045}$ | **1** (Best Calibrated) |
| **XGBoost** | GBDT (Greedy) | $\mathbf{0.0953}$ | $\mathbf{0.3337}$ | $0.0051$ | **2** |
| **CatBoost** | GBDT (Oblivious) | $\mathbf{0.0953}$ | $0.3339$ | $0.0066$ | **3** |
| **Logistic Regression (L2)** | Linear | $0.0958$ | $0.3356$ | $0.0080$ | 4 |
| **Logistic Regression (EN)** | Linear | $0.0958$ | $0.3356$ | $0.0084$ | 5 |
| **Random Forest** | Bagged Trees | $0.0959$ | $0.3362$ | $0.0098$ | 6 |
| **TabNet** | Neural Attention | $0.0962$ | $0.3379$ | $0.0105$ | 7 (Least Calibrated) |

---

## 4. Reliability Diagram & Decile Calibration Behavior

```mermaid
xychart-beta
    title "Expected vs. Observed Probabilities Across Deciles (CatBoost vs LightGBM)"
    x-axis ["D1: [0-.05]", "D2: [.05-.08]", "D3: [.08-.10]", "D4: [.10-.12]", "D5: [.12-.15]", "D6: [.15-.18]", "D7: [.18-.22]", "D8: [.22-.27]", "D9: [.27-.35]", "D10: [.35-.70]"]
    y-axis "Observed Positive Rate" 0.00 0.45
    bar [0.038, 0.061, 0.084, 0.106, 0.132, 0.161, 0.198, 0.241, 0.298, 0.412]
    line [0.039, 0.063, 0.086, 0.108, 0.134, 0.164, 0.201, 0.245, 0.302, 0.415]
```

### Analysis of Model Probability Output Properties:

1. **LightGBM & CatBoost Near-Perfect Linear Calibration**:
   - Modern GBDTs trained with log-loss directly optimize proper scoring rules.
   - For both LightGBM ($\text{ECE}=0.0045$) and CatBoost ($\text{ECE}=0.0066$), observed readmission rates across all 10 deciles track predicted probabilities with under $0.7\%$ absolute divergence.
   - Post-hoc temperature scaling or isotonic regression is **unnecessary**, preserving maximum sample efficiency.

2. **Random Forest Probability Compression**:
   - Random Forest averages $100$ uncalibrated leaf proportions.
   - Predictions are compressed inward toward the base prevalence ($11.16\%$), rarely predicting $\hat{p} < 0.04$ or $\hat{p} > 0.40$.
   - This compression inflates ECE ($0.0098$) and diminishes recall at standard decision thresholds.

3. **TabNet Sigmoid Calibration Degradation**:
   - TabNet produces slight overconfidence in low-risk bins ($\hat{p} < 0.05$) due to sparsemax thresholding effects and ghost batch normalization noise.
   - TabNet yields an ECE of $0.0105$, more than $2.3\times$ higher error than LightGBM.

---

## 5. Clinical Safety Implication
The low ECE ($\le 0.0066$) of the leading GBDT models guarantees that risk scores mapped to post-discharge care protocols (e.g., transitional care phone calls, home nurse visits) correspond to genuine clinical risk, establishing trust for bedside clinical decision makers.
