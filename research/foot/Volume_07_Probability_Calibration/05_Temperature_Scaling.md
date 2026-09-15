# Chapter 05 — Temperature Scaling Formulation & Results

## 5.1 Parameter Optimization & Formulation

Temperature Scaling optimizes a single scalar parameter $T > 0$ to scale logit vectors prior to applying the softmax function:

$$\hat{p}_i = \text{softmax}\left(\frac{z_i}{T}\right)$$

To guarantee $T > 0$ strictly during optimization, $T$ is parameterized via log-temperature: $T = \exp(\theta)$.

### Validation Parameter Fitting
- **Optimizer**: L-BFGS with line search
- **Loss Function**: Negative Log-Likelihood (Cross-Entropy) on validation set ($N=1,006$)
- **Initialization**: $T_0 = 1.0$ ($\theta_0 = 0.0$)
- **Fitted Parameter**: $T^* = 0.9860$ ($\theta^* = -0.0141$)

---

## 5.2 Performance Metrics Summary

Evaluating Temperature Scaling ($T^* = 0.9860$) across validation and held-out test sets:

| Split | Method | NLL ↓ | ECE ↓ | MCE ↓ | Brier ↓ | Accuracy | Macro F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation** | Raw Baseline | 0.8692 | 0.0541 | 0.1412 | 0.1130 | 0.6720 | 0.6705 |
| **Validation** | Temp Scaling ($T^*=0.9860$) | **0.8694** | **0.0543** | **0.1428** | **0.1130** | 0.6720 | 0.6705 |
| **Test** | Raw Baseline | 0.8779 | 0.0424 | 0.0753 | 0.1155 | 0.6690 | 0.6683 |
| **Test** | Temp Scaling ($T^*=0.9860$) | **0.8785** | **0.0388** | **0.2006** | **0.1155** | 0.6690 | 0.6683 |

---

## 5.3 Key Findings

1. **Test ECE Reduction**: Temperature Scaling slightly reduces test ECE from **0.0424** to **0.0388** ($\Delta = -0.0036$).
2. **Minimal Likelihood Shift**: Test NLL remains essentially unchanged ($0.8779 \to 0.8785$).
3. **Strict Class Rank Preservation**: Because scaling by $T^* > 0$ is a monotonic transformation, argmax predictions remain strictly identical ($\text{Accuracy} = 66.90\%$, $\text{Macro F1} = 0.6683$).
4. **Validation NLL Delta**: Because Temperature Scaling validation NLL ($0.8694$) is higher than Vector Scaling validation NLL ($0.8044$) by $0.0650$ ($> 0.005$), Vector Scaling is preferred.
