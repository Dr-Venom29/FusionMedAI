# Chapter 08 — Uncertainty Metric Selection & Configuration

## 1. Primary Metric Selection

Based on empirical held-out test evaluation ($N=1,006$):

1. **Primary Uncertainty Metric**: **Total Predictive Entropy ($H(\bar{p})$)**
   - **Error Detection AUROC**: **0.7291**
   - **Error Detection AUPRC**: **0.5391**
   - **Rationale**: Achieves the highest error-detection performance across held-out test samples, effectively identifying predictions where the model is likely to be incorrect.

2. **Secondary Uncertainty Metric**: **Predictive Variance ($\text{Var}(p)$)**
   - **Error Detection AUROC**: **0.6508**
   - **Error Detection AUPRC**: **0.4472**
   - **Rationale**: Captures class probability instability across stochastic passes.

3. **Empirical Pass Count**: **$N^{*} = 10$ MC Dropout Passes**
   - **Rationale**: Empirically selected via rule-based pass count convergence analysis ($\Delta H \le 10^{-3}$ nats, $\Delta \text{Var} \le 10^{-4}$).

---

## 2. Frozen Final Model Artifact Configuration

The deployment configuration is saved to `experiments/foot/final_model/uncertainty.json`:

```json
{
  "modality": "foot",
  "model": "efficientnet_b3",
  "checkpoint": "experiments/foot/architecture_benchmark/efficientnet_b3/checkpoints/best_model.pt",
  "calibration": "vector_scaling",
  "stochastic_passes_N": 10,
  "mean_predictive_variance": 0.000437,
  "mean_predictive_entropy": 0.8231,
  "mean_mutual_information": 0.0029,
  "error_detection_auroc_var": 0.6508,
  "error_detection_auprc_var": 0.4472
}
```
