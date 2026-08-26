# 9. Reproducibility and System Manifest

This chapter details the configuration, system environment, and data/checkpoint checksums required to replicate the experimental results documented in this volume.

---

## 9.1 Hardware and Operating System

The baseline experiments were executed on the following host configuration:
- **Operating System**: Microsoft Windows (Windows OS version `windows`)
- **Execution Target**: Central Processing Unit (CPU)
- **CUDA Device**: N/A (CPU-only execution verified)

---

## 9.2 Software Dependencies

The python environment was configured with the following package versions:
- **Python**: `3.12.2 (tags/v3.12.2:6abddd9)`
- **PyTorch**: `2.11.0+cpu` or compatible
- **Torchvision**: `0.16.0+cpu` or compatible
- **NumPy**: `1.26.4`
- **Pandas**: `2.2.1`
- **SciPy**: `1.12.0`
- **Scikit-Learn**: `1.4.1.post1`

---

## 9.3 Data Split and Checkpoint Hashes

### 9.3.1 Model Checkpoint
We evaluated the best pre-trained EfficientNet-B3 baseline model:
- **Checkpoint Location**: `experiments/efficientnet_b3/checkpoints/best_model.pt`
- **SHA-256 Checksum**: `263e8a3f9a9fb7aa995a8aafc40f13b9741d7da02889cf69b3168a9e569f17f2`

### 9.3.2 Test Dataset Split
We evaluated predictions on the frozen validation-test split:
- **CSV Split Path**: `datasets/processed/splits/test.csv`
- **Dataset Size**: $367$ retinal scans
- **SHA-256 Checksum**: `9ea965cf4502dfcbcc871f308a3fa38e6ff5a7f9859f13e73a0e5f98cf48c3b9` (re-computed from actual test split file)

---

## 9.4 Calibration Integration

Post-hoc Temperature Scaling calibration was restored from the validation experiment:
- **Calibration Run Directory**: `experiments/calibration/v004_temperature_scaling`
- **Calibrated Temperature ($T$)**: `1.6217584609985352`
- **Calibration State File**: `experiments/calibration/v004_temperature_scaling/calibration_state.pt`
- **Calibration Status**: `Verified` (hashes of evaluated checkpoint and calibrated model match exactly)

---

## 9.5 Uncertainty Pipeline Hyperparameters

The following parameters were utilized during uncertainty estimation:
- **Random Seed**: `42`
- **Number of MC Passes ($N$)**: `25`
- **Convergence Targets**: $N \in \{5, 10, 25, 50\}$
- **Dropout Active Layer**: `backbone.classifier.0`
- **Dropout Rate ($p$)**: `0.3`
- **BatchNorm Mode**: `Evaluation` (`.eval()`)
- **Weights Mode**: `Frozen` (no gradients computed)
