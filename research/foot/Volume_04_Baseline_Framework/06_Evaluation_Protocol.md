# 06 Evaluation Protocol — Phase 10.4

## Evaluation Directives

1. **Frozen Test Set Guarantee**: Test partition (1,006 images / 181 source groups) evaluated exactly once after configuration freeze.
2. **Deterministic Transforms**: 0% stochastic data augmentation during evaluation (Resize 224x224 + Observed Normalization).
3. **Primary Metric**: **Macro F1-Score**
4. **Secondary Metrics**: Top-1 Accuracy, Balanced Accuracy, Weighted F1, Class-wise Precision/Recall, $4 \times 4$ Confusion Matrix, One-vs-Rest ROC-AUC.
