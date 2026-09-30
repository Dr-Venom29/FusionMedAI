# Phase C4 — Baseline Modeling: Probability Calibration & Reliability (C4.8)

## 1. Clinical Importance of Probability Calibration

In medical decision support systems, discrimination (ROC-AUC) alone is insufficient. A model that assigns a predicted risk of $20\%$ to a patient must correspond to an actual observed empirical readmission rate of approximately $20\%$ among similar patients.

Calibration is formally audited using two primary statistical quantities:
1. **Brier Score**: Mean squared error between predicted probability $p_i$ and true binary label $y_i$.
2. **Log Loss (Cross-Entropy)**: Heavily penalizes confident incorrect probability estimates.
3. **Expected Calibration Error (ECE)**: Weighted average difference between predicted confidence and observed frequency across 10 uniform probability bins:
   $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

---

## 2. Calibration Behavior Across Baseline Architectures

- **Logistic Regression (L2 / ElasticNet)**:
  - Logistic regression directly optimizes log loss and often provides comparatively well-behaved probability estimates, but calibration is empirically assessed rather than assumed.
  - Empirical Test Brier Score: $0.0958$ | Log Loss: $0.3356$.
- **Random Forest**:
  - Tends to push probabilities toward the central empirical mean due to variance reduction across trees, rarely outputting extreme probabilities ($<0.05$ or $>0.60$).
  - Empirical Test Brier Score: $0.0959$ | Log Loss: $0.3362$.
- **Gradient Boosted Trees (XGBoost / LightGBM)**:
  - Generates sharper probability distributions with the lowest test log loss ($0.3338–0.3339$) and Brier score ($0.0953$).

---

## 3. Baseline Calibration Status & Post-Hoc Scope

> [!IMPORTANT]
> **Raw Uncalibrated Baselines**: No post-hoc probability calibration (such as Platt scaling, isotonic regression, or temperature scaling) was applied during Phase C4. All predictions and probability-dependent metrics represent raw, uncalibrated model outputs. Formal post-hoc calibration, reliability diagrams, and Platt/Isotonic comparison will be systematically evaluated in **Phase C7 (Probability Calibration)**.
