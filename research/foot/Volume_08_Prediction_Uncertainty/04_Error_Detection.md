# Chapter 04 — Misclassification Error Detection

## 1. Held-Out Test Evaluation Protocol

Using the frozen EfficientNet-B3 model, frozen Vector Scaling calibrator, and empirically selected $N^{*}=10$ MC Dropout pass count, uncertainty estimation was evaluated across the held-out test split ($N=1,006$ images).

Misclassification error detection measures the ability of uncertainty metrics to assign higher scores to incorrect predictions ($Y \neq \hat{Y}$) than correct predictions ($Y = \hat{Y}$).

---

## 2. Empirical Error Detection Performance

| Metric | Measure Purpose | AUROC | AUPRC | Baseline Error Rate | Rank | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Total Predictive Entropy ($H(\bar{p})$)** | Overall predictive uncertainty | **0.7291** | **0.5391** | 0.3231 | 1 | **PRIMARY** |
| **Predictive Variance ($\text{Var}(p)$)** | Prediction instability | **0.6508** | **0.4472** | 0.3231 | 2 | **SECONDARY** |
| **Mutual Information ($MI$)** | Epistemic model uncertainty | **0.6399** | **0.4424** | 0.3231 | 3 | Evaluated |

---

## 3. Findings

1. **Total Predictive Entropy** achieves the highest error-detection performance (**AUROC 0.7291**, **AUPRC 0.5391**), outperforming Predictive Variance (+0.0783 AUROC, +0.0919 AUPRC).
2. All three evaluated uncertainty metrics achieved AUROC values above 0.50 and AUPRC values above the test error-rate baseline of 0.3231, indicating discrimination above the corresponding reference levels in this evaluation.
3. Total Predictive Entropy provided the strongest error-detection performance among the three evaluated uncertainty measures. The result is consistent with uncertainty arising from prediction ambiguity, but the experiment does not isolate causal sources of error.
