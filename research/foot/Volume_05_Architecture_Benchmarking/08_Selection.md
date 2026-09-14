# 08 Selection — Final Backbone Decision

## Pre-Defined Model Selection Protocol

To maintain scientific integrity and prevent post-hoc metric selection bias, the decision criteria for selecting the winning Foot Ulcer backbone architecture are strictly frozen prior to inspecting benchmark results.

---

## Model Selection Hierarchy

### 1. Primary Selection Criterion
- **Highest Test Macro F1**:
  - Since all four Wagner grades (Grade 0, Grade 1, Grade 2, Grade 3/4) carry critical clinical weight, unweighted Macro F1 across 4 classes is the primary metric.
  - The winning candidate must demonstrate a statistically significant improvement over the ResNet-50 baseline (`Macro F1 = 0.6339`, 95% CI excluding zero).

### 2. Secondary Selection Criteria (Tie-Breakers)
If two or more candidate architectures demonstrate equivalent or statistically indistinguishable Macro F1 performance ($\Delta \text{Macro F1} < 0.01$ and overlapping 95% CIs), the winner is selected according to the following ordered secondary criteria:

1. **Balanced Accuracy**: Higher overall class-balanced recall across grades.
2. **Grade 2 ↔ Grade 3 Boundary Recall**: Higher sensitivity on Grade 3 (abscess / osteomyelitis), minimizing dangerous under-triaging to superficial Grade 2.
3. **Macro ROC-AUC**: Superior overall probabilistic separation.
4. **Computational Efficiency**: Smaller total parameter count ($M$) and lower inference latency ($ms$), favoring lighter architectures (e.g., EfficientNet over ViT) when accuracy is tied.

---

## Acceptance Gate Criteria

For a candidate architecture to be accepted as the new Foot Ulcer backbone in FusionMedAI, it must meet all of the following gate conditions:

1. **Baseline Outperformance**: Test Macro F1 > 0.6339 (ResNet-50 baseline).
2. **Reproducibility Gate**: Exact weight check and evaluation match across 2 independent runs under fixed seed (`42`).
3. **Checkpoint Verification Gate**: Checkpoint files (`best_model.pt`, `last_model.pt`, `config.json`) pass strict deserialization and dummy inference verification via `verify_checkpoints.py`.
4. **Zero Contamination**: Evaluation performed strictly on frozen test set (`test.csv`, N=1,006) with zero hyperparameter tuning post-unblinding.

---

## Winner Decision Matrix (Placeholder)

| Metric | ResNet-50 (Baseline) | Selected Backbone | Delta ($\Delta$) | Pass/Fail Gate |
| :--- | :---: | :---: | :---: | :---: |
| **Macro F1** | `0.6339` | TBD | TBD | Pending |
| **Balanced Accuracy** | `0.6391` | TBD | TBD | Pending |
| **Macro ROC-AUC** | `0.8423` | TBD | TBD | Pending |
| **Parameter Count** | `23.51 M` | TBD | TBD | Pending |
| **Inference Latency** | — | TBD | TBD | Pending |
