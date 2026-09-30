# Phase C4 — Baseline Modeling: LightGBM Classifier (C4.6)

## 1. Model Architecture & Rationale

LightGBM utilizes a leaf-wise (best-first) tree growth strategy rather than level-wise growth, enabling it to minimize loss more aggressively while maintaining high computational efficiency.

$$\text{Loss Reduction} = \max_{j, s} \left( \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right)$$

---

## 2. Conservative Baseline Hyperparameter Configuration

```json
{
  "model_family": "gradient_boosting",
  "name": "lightgbm",
  "n_estimators": 100,
  "max_depth": 5,
  "num_leaves": 31,
  "learning_rate": 0.05,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "random_state": 42,
  "verbose": -1,
  "n_jobs": -1
}
```

### Design Decisions:
- `num_leaves = 31` with `max_depth = 5`: Constrains tree complexity and limits leaf depth.
- `subsample = 0.8` & `colsample_bytree = 0.8`: Regularization against collinear clinical attributes.
