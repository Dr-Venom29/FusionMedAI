# 05 Training Protocol — Phase 10.4

## Execution & Convergence Behavior

- **Maximum Epochs**: 20
- **Early Stopping Trigger**: Stopped at epoch 11 (Patience = 10 on `val_loss`).
- **Best Epoch**: Epoch 1
- **Best Validation Loss**: `0.9264`
- **Best Validation Macro F1**: `0.6689`
- **Best Validation Accuracy**: `0.6730`
- **Observed Behavior**: Training loss dropped rapidly from epoch 1 to 11 while validation loss plateaued and increased, indicating early overfitting under the baseline training configuration.
