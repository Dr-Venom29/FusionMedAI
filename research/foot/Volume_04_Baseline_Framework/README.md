# Volume 04: Foot Ulcer Baseline Framework (Phase 10.4)

## Executive Summary

Volume 04 documents the baseline modeling framework, experimental contract freeze, model training procedures, checkpointing policies, and baseline evaluation benchmarks for the Diabetic Foot Ulcer (DFU) Wagner 4-class classification modality.

The baseline framework establishes an un-augmented/standardized benchmark using a single primary vision backbone (**EfficientNet-B0**) to evaluate classification performance on the leakage-controlled canonical modeling population (**10,050 images** across **1,770 source groups**).

---

## Volume Structure

1. **[01 Baseline Experimental Contract](file:///d:/FusionMedAI/research/foot/Volume_04_Baseline_Framework/01_Baseline_Experimental_Contract.md)**: Frozen parameters, task scope, dataset split paths, normalization statistics, optimizer settings, and evaluation metrics for Phase 10.4.

---

## Key Experimental Contract Directives

| Parameter | Value / Directives |
| :--- | :--- |
| **Task** | Wagner 4-Class Classification (`Grade 1`, `Grade 2`, `Grade 3`, `Grade 4`) |
| **Input Image** | $224 \times 224 \times 3$ RGB |
| **Primary Baseline Model** | `EfficientNet-B0` (Pre-trained ImageNet weights) |
| **Dataset Splits** | Train: 8,038 (80%) \| Val: 1,006 (10%) \| Test: 1,006 (10%) |
| **Normalization Mean** | `[0.4937, 0.3630, 0.3272]` (Observed RGB dataset mean) |
| **Normalization Std** | `[0.1745, 0.1632, 0.1551]` (Observed RGB dataset std) |
| **Train Augmentations** | Horizontal Flip ($p=0.5$), Rotation ($\pm 15^\circ$), Color Jitter ($b=0.2, c=0.2, s=0.1$) |
| **Val / Test Pipeline** | Resize to $224 \times 224$, Observed Normalization (100% Deterministic) |
| **Random Seed** | `42` |
| **Batch Size** | `32` |
| **Optimizer** | `AdamW` ($\text{lr} = 1\text{e-}4, \text{weight\_decay} = 1\text{e-}4$) |
| **Scheduler** | `CosineAnnealingLR` ($T_{\text{max}} = 20, \eta_{\text{min}} = 1\text{e-}6$) |
| **Epochs** | `20` epochs (Early stopping patience = `10` on `val_loss`) |
| **Loss Function** | Sqrt Inverse Frequency Weighted Cross-Entropy (`[1.0870, 1.0669, 1.0000, 1.0731]`) |
| **Primary Metric** | **Macro F1-Score** |
