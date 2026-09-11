# Phase 10.3.4 — Class Imbalance & Sampling Strategy Report

## 1. Class Distribution & Imbalance Ratio

The canonical modeling population comprises **10,050 images** across **1,770 source groups**.

| Wagner Grade | Clinical Description | Canonical Images | Image % | Source Groups | Group % |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Grade 1** | Superficial Ulcer | **3,091** | 30.76% | **491** | 27.74% |
| **Grade 2** | Deep Ulcer to Tendon/Bone | **2,752** | 27.38% | **466** | 26.33% |
| **Grade 3** | Deep Ulcer with Osteomyelitis/Abscess | **1,595** | 15.87% | **353** | 19.94% |
| **Grade 4** | Partial Foot Gangrene | **2,612** | 25.99% | **460** | 25.99% |
| **Total** | | **10,050** | **100.00%** | **1,770** | **100.00%** |

- **Imbalance Ratio**: **1.18:1** (Grade 1 vs Grade 3).
- **Classification**: Mild to moderate class imbalance. Grade 3 (15.87%) is underrepresented compared to Grade 1 (30.76%).

---

## 2. Partition Split Preservation

Group-stratified partitioning preserved class proportions across all three splits:

| Split | Grade 1 % | Grade 2 % | Grade 3 % | Grade 4 % |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 23.55% | 24.45% | 27.83% | 24.17% |
| **Val** | 23.56% | 24.45% | 27.83% | 24.16% |
| **Test** | 23.56% | 24.45% | 27.83% | 24.16% |

---

## 3. Loss Weighting Options

| Wagner Grade | Inverse Frequency Weights | Sqrt Inverse Frequency (Recommended) | Effective Number Weights (\beta=0.999) |
| :--- | :---: | :---: | :---: |
| **Grade 1 (Code 0)** | `1.0615` | `1.087` | `1.0128` |
| **Grade 2 (Code 1)** | `1.0226` | `1.0669` | `1.0039` |
| **Grade 3 (Code 2)** | `0.8983` | `1.0` | `0.9775` |
| **Grade 4 (Code 3)** | `1.0344` | `1.0731` | `1.0066` |

---

## 4. Modeling & Sampling Recommendations

1. **Sampling Strategy**: Use standard epoch-based random shuffling without heavy oversampling. Oversampling Grade 3 images risks memorizing specific patient source group artifacts.
2. **Loss Function**: Use Cross-Entropy Loss with Sqrt Inverse Frequency weights or Focal Loss ($\gamma = 2.0$) to mitigate mild minority class suppression.
3. **Metric Focus**: Macro F1-score and Grade 3 recall will serve as primary evaluation metrics during Phase 10.4+.
