# 07 Baseline Results — ResNet-50 Benchmark

## 1. Overall Test Scores (1,006 Images)

| Metric | Measured Score |
| :--- | :---: |
| **Macro F1-Score (Primary)** | **`0.6339`** (`63.39%`) |
| **Top-1 Accuracy** | **`0.6372`** (`63.72%`) |
| **Balanced Accuracy** | **`0.6391`** (`63.91%`) |
| **Weighted F1-Score** | **`0.6319`** (`63.19%`) |
| **Macro ROC-AUC** | **`0.8423`** |
| **Test Loss** | **`1.0548`** |

---

## 2. Class-Wise Metrics

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | ---: |
| **Grade 1** | `0.6386` | `0.5443` | `0.5877` | 237 |
| **Grade 2** | `0.7020` | `0.5650` | `0.6261` | 246 |
| **Grade 3** | `0.5684` | `0.5786` | `0.5735` | 280 |
| **Grade 4** | `0.6573` | **`0.8683`** | **`0.7482`** | 243 |

---

## 3. Test Set Confusion Matrix

```text
                 Predicted
              G1   G2   G3   G4

Actual G1    129   34   42   32
Actual G2     24  139   65   18
Actual G3     39   19  162   60
Actual G4     10    6   16  211
```

![Foot DFU Baseline — ResNet-50 Confusion Matrix](file:///d:/FusionMedAI/research/foot/Volume_04_Baseline_Framework/images/confusion_matrix.png)
