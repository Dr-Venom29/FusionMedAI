# 07 Model Comparison — Tradeoff Analysis

## 10.5.13 Confusion Analysis & Grade 2 ↔ Grade 3 Boundary

Phase 10.3 EDA identified Grade 2 $\leftrightarrow$ Grade 3 as the most difficult visual boundary due to overlapping ulcer depth and exudate features.

EfficientNet-B3 confusion matrix ($4 \times 4$):
```
[[128,  33,  52,  24],
 [ 13, 175,  42,  16],
 [ 35,  24, 194,  27],
 [ 27,   7,  33, 176]]
```

- **Grade 2 $\rightarrow$ Grade 3 confusion**: 42 images
- **Grade 3 $\rightarrow$ Grade 2 confusion**: 24 images
- **Grade 2 Sensitivity (Recall)**: 71.14% ($175 / 246$)
- EfficientNet-B3 substantially reduced Grade 2 under-triaging compared to baseline ResNet-50 and transformer candidate architectures.

---

## 10.5.14 Architecture-Specific Observations

### EfficientNet-B3
- Achieved highest Test Macro F1 (`0.6683`), Balanced Accuracy (`0.6672`), and Macro ROC-AUC (`0.8685`).
- Achieved strong Grade 2 performance (`F1 = 0.7216`).
- Provides the best overall predictive result under the pre-defined benchmark protocol.

### EfficientNet-B0
- Produced nearly identical performance (`Macro F1 = 0.6672`) while requiring only 4.01M parameters, 15.31 MB model size, and 37.89 ms batch latency ($2.25\times$ fewer parameters and $1.86\times$ faster than B3).
- Difference from B3 is only +0.0011 Macro F1.
- Retained as the lightweight efficiency alternative for resource-constrained deployments.

### ConvNeXt-Tiny
- Achieved Macro F1 = `0.6566` and Macro ROC-AUC = `0.8613`.
- Competitive CNN performance but required $2.6\times$ more parameters and $1.56\times$ more latency than B3 without exceeding its accuracy.

### Swin-Tiny
- Achieved Macro F1 = `0.6266` (below ResNet-50 baseline and both EfficientNets).
- Severe confusion on Grade 2 (`Grade 2 F1 = 0.5389`, 78 misclassifications to Grade 3). Not selected.

### ViT-B/16
- Lowest benchmark performance (`Macro F1 = 0.5788`, `Grade 2 F1 = 0.4309`).
- Highest computational cost (85.80M params, 327.31 MB, 310.28 ms latency). Not selected.
