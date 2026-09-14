# 01 Baseline Experimental Contract — Phase 10.4.1

## 1. Overview & Baseline Objective

Phase 10.4.1 explicitly defines and freezes the **Baseline Experimental Contract** for the Foot Ulcer modality.

The baseline model is initially treated strictly as a **four-class image classifier**:
- **Input**: Foot-ulcer RGB image ($224 \times 224 \times 3$)
- **Output**: Wagner severity grade:
  - `Grade 1` (Class Index `0`) — Superficial ulcer
  - `Grade 2` (Class Index `1`) — Deep ulcer without bone involvement
  - `Grade 3` (Class Index `2`) — Deep ulcer with abscess, osteomyelitis, or joint sepsis
  - `Grade 4` (Class Index `3`) — Localized gangrene

---

## 2. Strict Scope Exclusions (Prohibited in Baseline Phase)

To ensure scientific rigor and prevent benchmark inflation or premature complexity, the following techniques are **STRICTLY PROHIBITED** during baseline model development (Phase 10.4):
- ❌ **Multimodal Fusion**: No fusion with Retinal imaging or structured Clinical data.
- ❌ **Uncertainty Estimation**: No MC Dropout, predictive entropy, or mutual information calculations.
- ❌ **Probability Calibration**: No Temperature Scaling or Isotonic Regression post-processing.
- ❌ **Grad-CAM-driven Loss**: No attention loss regularization or saliency-guided optimization.
- ❌ **Out-of-Distribution (OOD) Detection**: No Mahalanobis or energy-based OOD scoring.
- ❌ **Clinical Features**: No patient tabular metadata inputs.
- ❌ **Special Ensembling Methods**: No multi-model averaging, snapshot ensembling, or stacking.

*All listed methods are assigned to downstream phases (Phases 10.5 through 10.9).*

---

## 3. Frozen Baseline Contract Specifications

The following parameters are frozen and immutable for all baseline experiments:

### 3.1 Dataset & Splits
- **Canonical Dataset Population**: 10,050 images across 1,770 source-image groups (`datasets/foot/processed/canonical_manifest.csv`).
- **Train Partition**: `datasets/foot/processed/splits/train.csv` (8,038 images / 1,412 groups)
- **Validation Partition**: `datasets/foot/processed/splits/val.csv` (1,006 images / 177 groups)
- **Test Partition**: `datasets/foot/processed/splits/test.csv` (1,006 images / 181 groups)
- **Split Index**: `datasets/foot/processed/splits/index.csv`
- **Group Leakage**: 0% source-group overlap across splits.

### 3.2 Preprocessing & Data Augmentation
- **Input Image Dimensions**: $224 \times 224 \times 3$ RGB.
- **Normalization Mean**: `[0.4937, 0.3630, 0.3272]` (Measured in Phase 10.3.1).
- **Normalization Std**: `[0.1745, 0.1632, 0.1551]` (Measured in Phase 10.3.1).
- **Train Augmentations**:
  - `RandomHorizontalFlip(p=0.5)`
  - `RandomRotation(degrees=15)`
  - `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)`
- **Val / Test Pipeline**: Resize to $224 \times 224$, Normalize to observed statistics. **0% Augmentation Leakage** (100% Deterministic).

### 3.3 Training Hyperparameters & Optimizer
- **Random Seed**: `42` (Enforced across Python, NumPy, PyTorch CPU & CUDA).
- **Batch Size**: `32`
- **Optimizer**: `AdamW` ($\text{lr} = 1\text{e-}4, \text{weight\_decay} = 1\text{e-}4$)
- **Learning Rate**: $1\text{e-}4$
- **Learning Rate Scheduler**: `CosineAnnealingLR` ($T_{\text{max}} = 20, \eta_{\text{min}} = 1\text{e-}6$)
- **Training Duration**: `20` epochs
- **Early Stopping**: Patience = `10` epochs based on Validation Loss (`val_loss`).
- **Primary Loss Function**: Sqrt Inverse Frequency Weighted Cross-Entropy (`[1.0870, 1.0669, 1.0000, 1.0731]`).
- **Baseline Variant Loss**: Standard Unweighted Cross-Entropy.

### 3.4 Model Checkpointing & Evaluation Metrics
- **Checkpoint Selection Criterion**: Lowest Validation Loss (`val_loss`); track best Validation Macro F1 (`val_macro_f1`).
- **Primary Metric**: **Macro F1-Score**
- **Secondary Metrics**:
  - Top-1 Accuracy
  - Weighted F1-Score
  - Class-wise Precision, Recall, F1-Score (specifically tracking Grade 3 Recall due to osteomyelitis severity)
  - Multi-Class $4 \times 4$ Confusion Matrix
  - One-vs-Rest ROC-AUC

---

## 4. Formal Acceptance Sign-Off

Phase 10.4.1 (Baseline Experimental Contract) is **FROZEN & SIGNED OFF**. Baseline model training code (`src/foot/model/`) must adhere strictly to this contract.
