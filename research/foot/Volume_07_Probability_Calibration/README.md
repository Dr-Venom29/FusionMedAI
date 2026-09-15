# Volume VII — Foot Ulcer Probability Calibration

> **Phase 10.7 Research Volume**  
> *Diabetic Foot Ulcer Probability Calibration for Frozen EfficientNet-B3 Model*

---

## Executive Summary

Phase 10.7 establishes post-hoc probability calibration for the primary Foot Ulcer classification model (**EfficientNet-B3**, frozen Phase 10.5 checkpoint). Raw deep learning classifiers output probabilities that require post-hoc alignment with empirical observed accuracy. This volume documents the fitting, evaluation, and selection of post-hoc calibrators (**Temperature Scaling** and **Vector Scaling**) fitted exclusively on validation logits ($N=1,006$) and evaluated on the held-out test set ($N=1,006$).

**Vector Scaling** ($w^* = [1.0410, 1.0430, 0.8711, 1.1301], b^* = [0.0292, 0.0625, 0.0630, -0.1547]$) was selected as the canonical calibration method, reducing Expected Calibration Error (ECE) on the held-out test set by **26.18%** (from $0.0424$ to $0.0313$) and Negative Log-Likelihood (NLL) from $0.8779$ to $0.8749$, while improving held-out test classification performance (Macro F1 = 0.6758 vs 0.6683).

---

## Volume Index & Sitemap

1. [Chapter 01 — Objectives & Research Position](01_Objectives.md)
2. [Chapter 02 — Calibration Method Formulations](02_Calibration_Methods.md)
3. [Chapter 03 — 10-Bin Equal-Width Evaluation Protocol](03_Evaluation_Protocol.md)
4. [Chapter 04 — Raw Model Calibration Diagnostics](04_Raw_Model_Calibration.md)
5. [Chapter 05 — Temperature Scaling Formulation & Results](05_Temperature_Scaling.md)
6. [Chapter 06 — Vector Scaling Formulation & Results](06_Vector_Scaling.md)
7. [Chapter 07 — Comparative Held-Out Test Results](07_Comparative_Results.md)
8. [Chapter 08 — Per-Grade Confidence Calibration Diagnostics](08_Classwise_Analysis.md)
9. [Chapter 09 — Final Calibrator Selection Rationale](09_Selection.md)
10. [Chapter 10 — Acceptance Criteria Compliance Checklist](10_Acceptance.md)

---

## Key Experimental Results

| Method | NLL ↓ | ECE ↓ | MCE ↓ | Brier ↓ | Accuracy | Macro F1 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Uncalibrated (Raw)** | 0.8779 | 0.0424 | 0.0753 | 0.1155 | 0.6690 | 0.6683 | Baseline |
| **Temperature Scaling ($T^*=0.9860$)** | 0.8785 | 0.0388 | 0.2006 | 0.1155 | 0.6690 | 0.6683 | Evaluated |
| **Vector Scaling** | **0.8749** | **0.0313** | **0.0895** | **0.1154** | **0.6769** | **0.6758** | **SELECTED** |

---

## Artifact Locations

- **Source Code**: [`src/foot/calibration/`](file:///d:/FusionMedAI/src/foot/calibration/)
- **Experiment Artifacts**: [`experiments/foot/calibration/`](file:///d:/FusionMedAI/experiments/foot/calibration/)
- **Final Selected Artifacts**: [`experiments/foot/final_model/calibration.json`](file:///d:/FusionMedAI/experiments/foot/final_model/calibration.json)
- **Automated Verification**: [`verification/foot/model/verify_calibration.py`](file:///d:/FusionMedAI/verification/foot/model/verify_calibration.py)
