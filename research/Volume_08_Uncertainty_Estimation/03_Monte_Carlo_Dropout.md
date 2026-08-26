# 3. Monte Carlo Dropout: Architecture and Validation

Implementing Monte Carlo (MC) Dropout requires verifying that the target model architecture contains suitable stochastic layers and that enabling them at inference generates meaningful variance without corrupting other components (like Batch Normalization). This chapter documents the inspection, layer discovery, and validation procedure.

---

## 3.1 EfficientNet-B3 Model Inspection

We programmatically inspected the instantiated `EfficientNetB3` classifier wrapper loading weights from `best_model.pt`. The architecture features the standard EfficientNet-B3 backbone from `torchvision.models` with a custom linear classification head:

- **Backbone**: EfficientNet-B3
- **Feature Extractor**: Convolutional blocks ending with `conv_head` (1536 channels) and `avgpool`.
- **Classification Head (`model.backbone.classifier`)**:
  - `backbone.classifier[0]`: `torch.nn.modules.dropout.Dropout` layer (default dropout rate $p = 0.3$).
  - `backbone.classifier[1]`: `nn.Linear` mapping 1536 features to $K = 5$ classes.

The discovery scan confirmed that exactly one dropout layer exists in the model:

```python
[('backbone.classifier.0', <class 'torch.nn.modules.dropout.Dropout'>)]
```

---

## 3.2 Selective Dropout Activation vs. BatchNorm

A common pitfall in MC Dropout is using `model.train()` to activate dropout. This is mathematically incorrect because `model.train()` also puts Batch Normalization (BatchNorm) layers in training mode. 

During normal training, BatchNorm layers track running statistics (mean and variance) of the batches. If BatchNorm is set to `train()` during inference, it will recompute these statistics on the test batches (or single images), leading to:
- Instability of predicted probabilities due to batch size dependency (especially under small batches or single-image inference).
- Contamination of the frozen calibration baseline.
- Leakage of batch-level statistics across test samples.

To avoid this, we implement **selective dropout activation**:

```python
def enable_only_dropout(model: nn.Module):
    # 1. Put the entire model in evaluation mode (freezes BatchNorm and weights)
    model.eval()
    
    # 2. Specifically put only nn.Dropout modules into training mode
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            module.train() # Activates dropout forward pass stochasticity
```

This guarantees that:
- $\text{Dropout} \to \text{ACTIVE}$
- $\text{BatchNorm} \to \text{EVAL}$
- $\text{Model Weights} \to \text{FROZEN}$

---

## 3.3 Mandatory Stochasticity Validation Check

To prevent executing a silent, pseudo-stochastic pipeline (where passes are identical due to layers being set incorrectly), we enforce a mandatory sanity check prior to test set inference.

### 3.3.1 Validation Procedure
1. Extract a small sample of $M = 5$ images from the test split.
2. Run $5$ stochastic passes.
3. Compute the raw logits ($z_t$) and calibrated probabilities ($p_t$) for each pass.
4. Calculate standard metrics of dispersion across the $5$ passes:
   - Mean logit standard deviation: $\text{std}(z_t)$
   - Mean probability standard deviation: $\text{std}(p_t)$
   - Mean probability variance across passes: $\text{Var}(p_t)$
5. **Fail-Fast Assertion**: Assert that $\text{std}(z_t) > 10^{-6}$ and $\text{Var}(p_t) > 10^{-7}$. If these thresholds are not met, the script terminates immediately with a runtime error.

### 3.3.2 validation Report Structure
The validation report is saved to `results/uncertainty/mc_dropout_validation.json` with the following schema:

```json
{
    "dropout_detected": true,
    "dropout_layers": [
        ["backbone.classifier.0", 0.3]
    ],
    "passes_tested": 5,
    "mean_logits_std": 0.28456,
    "mean_probability_std": 0.04123,
    "logits_differ": true,
    "probabilities_differ": true,
    "probability_variance_nonzero": true,
    "aggregate_variance": 0.00284,
    "timestamp": 1785859210.42
}
```

This verify that the dropout layers generate active, non-zero stochastic variation in the model's outputs.
