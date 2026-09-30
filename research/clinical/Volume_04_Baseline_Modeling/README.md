# Research Volume 04: Baseline Modeling Framework & Evaluation

## Overview
This volume documents the formal construction, training, threshold analysis, probability calibration, subgroup error stratification, and performance benchmarking of the four core tabular machine learning baselines for the Clinical Modality of **FusionMedAI**:
1. **Regularized Logistic Regression** (L2 and ElasticNet penalties)
2. **Random Forest Classifier** (Bootstrap Aggregation Ensemble)
3. **XGBoost Classifier** (Exact Greedy Gradient Boosted Trees)
4. **LightGBM Classifier** (Leaf-Wise Gradient Boosted Trees)

All models are trained strictly on the frozen Phase C2 training partition ($69,519$ encounters) using the frozen Phase C3 feature representation contract ($119$ feature dimensions), with zero leakage or parameter estimation on validation or test sets.

---

## Volume Contents

- [01_modeling_contract.md](01_modeling_contract.md): Frozen modeling contract, inputs, random seed, and evaluation metric hierarchy.
- [02_preprocessing.md](02_preprocessing.md): Feature transformation pipeline architecture, encoding standards, and 119-dimension accounting.
- [03_logistic_regression.md](03_logistic_regression.md): Regularized linear baselines (L2 / ElasticNet) specifications and training dynamics.
- [04_random_forest.md](04_random_forest.md): Random Forest ensemble bagging baseline configuration and regularizations.
- [05_xgboost.md](05_xgboost.md): XGBoost gradient boosted tree baseline configuration and tree depth parameters.
- [06_lightgbm.md](06_lightgbm.md): LightGBM leaf-wise boosting baseline architecture and hyperparameters.
- [07_threshold_analysis.md](07_threshold_analysis.md): Clinical decision threshold sweep across $\theta \in \{0.10, 0.20, 0.30, 0.40, 0.50\}$.
- [08_calibration_analysis.md](08_calibration_analysis.md): Reliability diagrams, Brier score, log loss, and Expected Calibration Error (ECE).
- [09_error_analysis.md](09_error_analysis.md): Subgroup error analysis (False Positives and False Negatives stratified across age, race, gender, diagnosis, and prior utilization).
- [10_baseline_comparison.md](10_baseline_comparison.md): Comprehensive cross-model performance synthesis and validation/test benchmarks.

---

## Phase C4 Verification Summary

```
==================================================
FusionMedAI: Phase C4 Baseline Verification Gate
==================================================

[ 1] Train split exists                   PASS
[ 2] Validation split exists              PASS
[ 3] Test split exists                    PASS
[ 4] Expected row counts verified         PASS
[ 5] Target encoding verified             PASS
[ 6] Feature contract loaded              PASS
[ 7] Identifier exclusion verified        PASS
[ 8] Train-only preprocessing fit         PASS
[ 9] Unknown-category handling            PASS
[10] No train/val/test overlap            PASS
[11] Logistic Regression trained          PASS
[12] Random Forest trained                PASS
[13] XGBoost trained                      PASS
[14] LightGBM trained                     PASS
[15] Required metrics generated           PASS
[16] Prediction artifacts generated       PASS
[17] Model configs generated              PASS
[18] Frozen artifact reproducibility      PASS
--------------------------------------------------
C4 STATUS                                PASS
==================================================
```
