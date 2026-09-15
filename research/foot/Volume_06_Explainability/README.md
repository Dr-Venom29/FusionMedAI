# Foot Ulcer Module — Phase 10.6
# Explainability & Visual Attribution Protocol

## Volume Overview

This volume documents Phase 10.6 of the Foot Ulcer Classification module within the FusionMedAI framework. The primary objective is to scientifically evaluate whether the selected **EfficientNet-B3** classification model bases its four-class Wagner grade predictions on anatomically relevant and visually meaningful wound regions (ulcer bed, ulcer margin, necrotic tissue) rather than background noise, border artifacts, or image-capture shortcuts.

---

## Chapter Structure

| Chapter | Title | Primary Focus |
| :--- | :--- | :--- |
| **[01 Objectives](01_Objectives.md)** | Research Question & Goals | Core research question, hypotheses, and scope |
| **[02 Explainability Method](02_Explainability_Method.md)** | Method Formulation & Target Layer | Grad-CAM mathematical formulation & layer verification (`backbone.features[8]`) |
| **[03 Evaluation Protocol](03_Evaluation_Protocol.md)** | Evaluation Protocol | Evaluation protocol & dataset-wide attribution concentration design |
| **[04 Qualitative Results](04_Qualitative_Results.md)** | Qualitative Visual Analysis | Visual examination across Wagner Grades 1–4 and visual attribution cases |
| **[05 Quantitative Results](05_Quantitative_Results.md)** | Quantitative Metrics | Empirical evaluation metrics (Area fractions, mass concentration, sanity checks) |
| **[06 Error Analysis](06_Error_Analysis.md)** | Misclassification Diagnosis | Visual attribution breakdown for false positives/negatives (e.g., Grade 2 ↔ Grade 3) |
| **[07 Acceptance](07_Acceptance.md)** | Acceptance Criteria & Summary | Phase 10.6 sign-off and transition to Phase 10.7 (Calibration) |

---

## Model Under Explanation

- **Architecture**: `EfficientNet-B3`
- **Frozen Checkpoint**: `experiments/foot/architecture_benchmark/efficientnet_b3/checkpoints/best_model.pt`
- **Selection Metric**: Test Macro F1 (`0.6683`), Balanced Accuracy (`0.6672`), Macro ROC-AUC (`0.8685`)
- **Target Feature Layer**: `backbone.features[8]` (Output shape: `[B, 1536, 7, 7]`)
- **Methodological Limitation**: Attribution Visualization $\neq$ Lesion Segmentation $\neq$ Clinical Validation
