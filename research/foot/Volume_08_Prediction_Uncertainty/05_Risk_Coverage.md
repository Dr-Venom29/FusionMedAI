# Chapter 05 — Selective Prediction & Risk-Coverage Analysis

## 1. Risk-Coverage Formulation

Selective prediction allows clinical decision-support systems to abstain from making automated predictions when uncertainty exceeds a threshold $\tau$. Samples with uncertainty below $\tau$ are retained, while samples with uncertainty above $\tau$ are flagged for expert manual review.

The Risk-Coverage relationship evaluates how the error rate (risk) on retained predictions drops as coverage is reduced by sorting samples in order of ascending uncertainty.

---

## 2. Discrete Coverage Threshold Tabulation

Sorting test samples ($N=1,006$) by ascending **Predictive Variance**:

| Coverage Level (%) | Retained Samples | Rejected Samples | Error Rate / Risk (%) | Accuracy (%) | Error Reduction |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 100.0% | 1,006 | 0 | 32.31% | 67.69% | Baseline |
| 90.0% | 905 | 101 | 27.40% | 72.60% | -4.91% |
| 80.0% | 805 | 201 | 22.86% | 77.14% | -9.45% |
| 70.0% | 704 | 302 | 18.32% | 81.68% | -13.99% |
| 60.0% | 604 | 402 | 14.24% | 85.76% | -18.07% |
| 50.0% | 503 | 503 | 11.20% | 88.80% | -21.11% |

---

## 3. Analysis & Key Observations

1. **Monotonic Risk Reduction**: As coverage decreases from 100% to 50%, observed test risk decreases from 32.31% to 11.20%, corresponding to a 21.11 percentage-point absolute reduction.
2. **Selective Performance**: At 80% coverage (rejecting the 20% most uncertain cases), accuracy improves from 67.69% to 77.14%.
3. The rejection ordering provides a selective-prediction mechanism in which higher-uncertainty samples are preferentially excluded from the retained prediction set.
