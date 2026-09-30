# Chapter 03 — Evaluation Protocol & Decision Hierarchy

## 3.1 Binning Protocol

All calibration diagnostics use a **fixed 10 equal-width bin protocol** across the confidence interval $[0.0, 1.0]$:

$$
\text{Bin}_m = \left[ \frac{m-1}{10}, \frac{m}{10} \right), \quad m = 1, \dots, 10
$$

The same 10 bins are applied identically across Raw Softmax, Temperature Scaling, and Vector Scaling to ensure fair and rigorous comparison.

---

## 3.2 Metric Definitions

### 3.2.1 Negative Log-Likelihood (NLL)

Primary likelihood metric (lower is better):

$$
\text{NLL} = -\frac{1}{N} \sum_{i=1}^N \log \hat{p}(y_i \mid x_i)
$$

### 3.2.2 Expected Calibration Error (ECE)

Primary calibration error metric (lower is better):

$$
\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|
$$

### 3.2.3 Maximum Calibration Error (MCE)

Worst-bin calibration deviation (lower is better):

$$
\text{MCE} = \max_{m=1,\dots,M} |\text{acc}(B_m) - \text{conf}(B_m)|
$$

### 3.2.4 Brier Score

Proper scoring rule for probability accuracy (lower is better):

$$
\text{BS} = \frac{1}{N} \sum_{i=1}^N \sum_{c=1}^K (\hat{p}_{i,c} - y_{i,c})^2
$$

### 3.2.5 Performance Preservation Check

Evaluates whether the selected calibrator preserves classification performance across:
- **Macro F1**
- **Balanced Accuracy**
- **Macro ROC-AUC**

*Changes in Macro F1, Balanced Accuracy, and Macro ROC-AUC are reported descriptively and are not used as the primary selection criterion.*

---

## 3.3 Method Selection Hierarchy

The final calibration method is selected according to the following decision rules:

1. **Primary Selection Criterion**: Lower validation NLL and ECE reduction relative to raw softmax.
2. **Parsimony Principle**: If Temperature Scaling achieves validation NLL within $0.005$ of Vector Scaling ($\text{NLL}_{\text{temp}} \le \text{NLL}_{\text{vec}} + 0.005$), **Temperature Scaling is selected** due to model parsimony ($1$ parameter vs $8$ parameters) and lower risk of validation overfitting. Otherwise, Vector Scaling is selected.
3. **Performance Preservation Check**: Report whether the selected calibrator preserves classification performance; changes in Macro F1, Balanced Accuracy, and Macro ROC-AUC are reported descriptively and are not used as the primary selection criterion.
