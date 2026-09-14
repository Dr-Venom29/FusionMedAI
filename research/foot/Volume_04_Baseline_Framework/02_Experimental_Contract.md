# 02 Experimental Contract — Phase 10.4

## 1. Frozen Baseline Parameters

- **Task Scope**: 4-Class Wagner Classification (`Grade 1`, `Grade 2`, `Grade 3`, `Grade 4`)
- **Input Image Dimensions**: $224 \times 224 \times 3$ RGB
- **Canonical Dataset Population**: 10,050 images across 1,770 source-image groups
- **Splits**:
  - Train: 8,038 images (1,412 groups)
  - Val: 1,006 images (177 groups)
  - Test: 1,006 images (181 groups)
- **Observed Normalization Statistics**:
  - Mean: `[0.4937, 0.3630, 0.3272]`
  - Std: `[0.1745, 0.1632, 0.1551]`
- **Random Seed**: `42`
- **Batch Size**: `32`
- **Optimizer**: `AdamW` ($\text{lr} = 1\text{e-}4$, $\text{weight\_decay} = 1\text{e-}4$)
- **Scheduler**: `CosineAnnealingLR` ($T_{\text{max}} = 20$, $\eta_{\text{min}} = 1\text{e-}6$)
- **Loss Function**: Unweighted CrossEntropy Loss
- **Evaluation Metric**: **Macro F1-Score** (Primary Benchmark)

---

## 2. Strict Scope Exclusions

Prohibited during baseline phase:
- ❌ Multimodal Fusion
- ❌ Uncertainty Estimation
- ❌ Post-hoc Calibration
- ❌ Grad-CAM Loss Regularization
- ❌ OOD Detection
- ❌ Clinical Tabular Features
- ❌ Model Ensembling
