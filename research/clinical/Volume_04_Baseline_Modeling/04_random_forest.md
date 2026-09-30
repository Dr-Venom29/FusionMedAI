# Phase C4 — Baseline Modeling: Random Forest Classifier (C4.4)

## 1. Model Architecture & Rationale

Random Forest provides an ensemble bagging baseline capable of capturing non-linear feature interactions (such as the interaction between insulin dose adjustments and acute glycemic test results) without requiring manual interaction engineering.

$$\hat{P}(y=1 \mid \mathbf{x}) = \frac{1}{B} \sum_{b=1}^B T_b(\mathbf{x})$$

Where each tree $T_b$ is trained on a bootstrap sample of the training dataset with random feature subspace sampling at each split.

---

## 2. Conservative Baseline Hyperparameter Configuration

```json
{
  "model_family": "ensemble_bagging",
  "name": "random_forest",
  "n_estimators": 100,
  "max_depth": 12,
  "min_samples_leaf": 20,
  "max_features": "sqrt",
  "bootstrap": true,
  "random_state": 42,
  "n_jobs": -1
}
```

### Regularization Controls:
- `max_depth = 12`: Prevents deep tree memorization on high-cardinality categorical dummy features.
- `min_samples_leaf = 20`: Restricts terminal leaf predictions to statistically stable patient sub-populations.
- `max_features = 'sqrt'`: Limits split candidates to $\approx \sqrt{119} \approx 11$ features per node, reducing inter-tree correlation.
