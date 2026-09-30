# Phase C4 — Baseline Modeling: Logistic Regression (C4.3)

## 1. Model Architecture & Rationale

Regularized Logistic Regression serves as the primary interpretable linear baseline for clinical tabular prediction. It maps linear combinations of scaled clinical features to predicted log-odds:

$$P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

Two regularization configurations are evaluated:
1. **L2 (Ridge)**: Controls multicollinearity across correlated clinical features (e.g., prior utilization and hospital length of stay).
2. **ElasticNet**: Combines L1 sparsity with L2 stability ($\ell_1\text{-ratio} = 0.50$), performing continuous feature selection across sparse one-hot categories.

---

## 2. Experimental Configuration

```json
{
  "model_family": "linear_model",
  "configurations": [
    {
      "name": "logistic_regression_l2",
      "penalty": "l2",
      "C": 1.0,
      "solver": "lbfgs",
      "max_iter": 1000,
      "random_state": 42
    },
    {
      "name": "logistic_regression_elasticnet",
      "penalty": "elasticnet",
      "C": 1.0,
      "l1_ratio": 0.5,
      "solver": "saga",
      "max_iter": 500,
      "random_state": 42
    }
  ]
}
```

---

## 3. Training & Convergence Profile

- **Feature Matrix**: Standardized continuous features + One-Hot encoded categoricals ($D = 119$ dimensions).
- **Optimization**: L-BFGS converges smoothly within $< 150$ iterations; SAGA converges within $500$ iterations.
- **Class Weighting**: Kept at natural distribution ($1.0 : 1.0$) to establish an unweighted baseline calibration anchor.
