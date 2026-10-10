# Volume VIII — Foot Ulcer Prediction Uncertainty Estimation

> **Phase 10.8 Research Volume**  
> *Diabetic Foot Ulcer Prediction Uncertainty Estimation for Frozen Calibrated EfficientNet-B3 Model*

---

## Executive Summary

Phase 10.8 establishes stochastic Monte Carlo (MC) Dropout uncertainty estimation for the primary Foot Ulcer classification model (**EfficientNet-B3**, frozen Phase 10.5 checkpoint) integrated with the frozen Phase 10.7 **Vector Scaling** calibrator. Using empirically selected $N^{*}=10$ stochastic passes under the calibrated Option B pipeline, the framework evaluates Predictive Mean ($\bar{p}$), Predictive Variance ($\text{Var}(p)$), Total Predictive Entropy ($H(\bar{p})$), Aleatoric Uncertainty ($\mathbb{E}[H(p_t)]$), and Epistemic Uncertainty / Mutual Information ($MI$).

The framework evaluates misclassification detection performance via Area Under the ROC Curve (AUROC) and Area Under the Precision-Recall Curve (AUPRC), disaggregates uncertainty across Wagner grades (G1, G2, G3, G4), inspects Grade 2 $\leftrightarrow$ Grade 3 boundary misclassifications, connects uncertainty to explainability via deterministic Grad-CAM heatmaps, and verifies selective prediction via risk-coverage rejection.

---

## Volume Index & Sitemap

1. [Chapter 01 — Objectives & Research Position](01_Objectives.md)
2. [Chapter 02 — Uncertainty Formulations & Method](02_Uncertainty_Method.md)
3. [Chapter 03 — MC Dropout Pass Count Convergence](03_MC_Dropout_Convergence.md)
4. [Chapter 04 — Misclassification Error Detection](04_Error_Detection.md)
5. [Chapter 05 — Selective Prediction & Risk-Coverage Analysis](05_Risk_Coverage.md)
6. [Chapter 06 — Classwise Uncertainty Analysis](06_Classwise_Analysis.md)
7. [Chapter 07 — Qualitative High-Uncertainty & Explainability Error Analysis](07_Error_Analysis.md)
8. [Chapter 08 — Uncertainty Metric Selection & Configuration](08_Selection.md)
9. [Chapter 09 — Phase 10.8 Acceptance Criteria & 12-Point Verification](09_Acceptance.md)

---

## Key Experimental Summary

| Metric | Measure Purpose | Error Detection AUROC | Error Detection AUPRC | Baseline Error Rate | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Total Predictive Entropy ($H(\bar{p})$)** | Overall predictive uncertainty | **0.7291** | **0.5391** | 0.3231 | **PRIMARY** |
| **Predictive Variance ($\text{Var}(p)$)** | Prediction instability | **0.6508** | **0.4472** | 0.3231 | **SECONDARY** |
| **Mutual Information ($MI$)** | Epistemic model uncertainty | **0.6399** | **0.4424** | 0.3231 | Evaluated |

---

## Artifact Locations

- **Source Code**: [`src/foot/uncertainty/`](../../../src/foot/uncertainty)
- **Experiment Output**: `experiments/foot/uncertainty/`
- **Frozen Final Manifest**: `experiments/foot/final_model/uncertainty.json`
- **12-Point Automated Verification Suite**: [`verification/foot/model/verify_uncertainty.py`](../../../verification/foot/model/verify_uncertainty.py)
