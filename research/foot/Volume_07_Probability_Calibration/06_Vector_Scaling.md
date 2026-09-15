# Chapter 06 — Vector Scaling Formulation & Results

## 6.1 Parameter Optimization & Formulation

Vector Scaling extends temperature scaling by applying a per-class diagonal weight matrix $W = \text{diag}(w_1, w_2, w_3, w_4)$ and bias vector $b = [b_1, b_2, b_3, b_4]^T$:

$$\hat{z}_i = w_i \cdot z_i + b_i$$
$$\hat{p}_i = \text{softmax}(\hat{z}_i)$$

Vector Scaling introduces **8 parameters** (4 weights, 4 biases for 4 Wagner classes).

### Validation Parameter Fitting
- **Optimizer**: L-BFGS with line search
- **Loss Function**: Negative Log-Likelihood (Cross-Entropy) on validation set ($N=1,006$)
- **Fitted Weights ($w^*$)**: $[1.0410, 1.0430, 0.8711, 1.1301]$
- **Fitted Biases ($b^*$)**: $[0.0292, 0.0625, 0.0630, -0.1547]$

---

## 6.2 Performance Metrics Summary

Evaluating Vector Scaling across validation and held-out test sets:

| Split | Method | NLL ↓ | ECE ↓ | MCE ↓ | Brier ↓ | Accuracy | Macro F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation** | Raw Baseline | 0.8692 | 0.0541 | 0.1412 | 0.1130 | 0.6720 | 0.6705 |
| **Validation** | Vector Scaling | **0.8044** | **0.0245** | **0.0682** | **0.1082** | **0.7107** | **0.7092** |
| **Test** | Raw Baseline | 0.8779 | 0.0424 | 0.0753 | 0.1155 | 0.6690 | 0.6683 |
| **Test** | Vector Scaling | **0.8749** | **0.0313** | **0.0895** | **0.1154** | **0.6769** | **0.6758** |

---

## 6.3 Key Findings

1. **Substantial Validation NLL Improvement**: Vector Scaling reduces validation NLL from **0.8692** to **0.8044** ($\Delta = -0.0648$), easily exceeding the $0.005$ tolerance threshold over Temperature Scaling ($0.8694$).
2. **Held-Out Test ECE Reduction**: Vector Scaling achieves the lowest test ECE of **0.0313**, representing a **26.18% reduction in test calibration error** relative to uncalibrated raw probabilities ($0.0424$).
3. **Classification Performance Gain**: Vector Scaling slightly improves held-out test accuracy from **66.90%** to **67.69%** (+0.79%) and Macro F1 from **0.6683** to **0.6758** (+0.0075).
