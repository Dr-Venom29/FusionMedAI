# 05 Evaluation Protocol — Phase 10.5

## 10.5.12 Evaluation Directives & Test-Set Isolation

1. **Untouched Test Set Policy**: The frozen test set ($N=1,006$ images / 181 source groups) remained completely untouched during model training and early stopping decisions.
2. **Post-Training Evaluation Order**:
   $$\text{Train (20 Epochs)} \longrightarrow \text{Save Checkpoints} \longrightarrow \text{Load Best } \texttt{val\_loss} \text{ Checkpoint} \longrightarrow \text{Single Test Evaluation}$$
3. **Primary Benchmark Metric**: **Test Macro F1-Score** across all 4 Wagner classes.
4. **Checkpoint Selection Criterion**: Checkpoints were selected strictly using minimum validation loss (`val_loss`). Validation Macro F1 is reported for reference and was not used for model selection.
5. **Bootstrap Confidence Intervals**: Image-level 95% Bootstrap Confidence Intervals ($B=1,000$ resamples, seed `42`) calculated for Macro F1 and Balanced Accuracy to quantify metric sampling uncertainty.
6. **Error Analysis & Confusion Matrices**: Quality-stratified misclassification error logs exported to `error_analysis.csv` and $4 \times 4$ confusion matrix heatmaps rendered for each architecture.

---

## 10.5.15 Overfitting Behavior & Early Stopping Analysis

The evaluation protocol revealed rapid overfitting on vision transformer architectures under the standard protocol:
- **EfficientNet-B3**: Reached best validation-loss checkpoint at epoch 1 (`val_loss = 0.8779`).
- **Swin-Tiny**: Reached best validation-loss checkpoint at epoch 2 (`val_loss = 1.5243`).
- **ViT-B/16**: Reached best validation-loss checkpoint at epoch 1 (`val_loss = 1.2080`).

Continued training beyond these early epochs reduced training loss further but increased validation loss, reinforcing the necessity of strict validation-loss checkpoint selection and early stopping rather than training to convergence on training loss.
