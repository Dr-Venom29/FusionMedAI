# Phase C4 — Baseline Modeling: XGBoost Classifier (C4.5)

## 1. Model Architecture & Rationale

XGBoost is an exact-split gradient boosted decision tree (GBDT) framework. It sequentially minimizes a regularized logistic loss objective using second-order Taylor expansions:

$$\mathcal{L}^{(t)} \approx \sum_{i=1}^N \left[ g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i) \right] + \Omega(f_t)$$

Where $g_i$ and $h_i$ are the first and second gradients of the log-loss loss function, and $\Omega(f_t) = \gamma T + \frac{1}{2}\lambda \sum w_j^2$ enforces structural complexity penalties.

---

## 2. Conservative Baseline Hyperparameter Configuration

```json
{
  "model_family": "gradient_boosting",
  "name": "xgboost",
  "n_estimators": 100,
  "max_depth": 5,
  "learning_rate": 0.05,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "eval_metric": "logloss",
  "random_state": 42,
  "n_jobs": -1
}
```

### Design Decisions:
- `learning_rate = 0.05`: Conservative shrinkage rate ensuring smooth convergence without aggressive overfitting.
- `subsample = 0.8` & `colsample_bytree = 0.8`: Stochastic row and feature subsampling preventing over-reliance on dominant utilization counts.
- `max_depth = 5`: Restricts tree depth to 5-way feature interaction capacity.
