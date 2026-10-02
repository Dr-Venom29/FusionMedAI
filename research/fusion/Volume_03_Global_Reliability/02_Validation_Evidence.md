# 02 Validation Evidence & Cohort Provenance — Phase C11.3

## 1. Modality Validation Cohorts

To ensure zero data snooping and complete experimental reproducibility, the evidence used for computing $R_i$ is drawn strictly from the canonical validation splits of each modality:

| Modality ($i$) | Primary Task | Validation Partition Source | Sample Count ($N_{\text{val}}$) | Frozen Calibration Checkpoint |
| :--- | :--- | :--- | :---: | :--- |
| **Retina ($R$)** | 5-Class Diabetic Retinopathy | `datasets/retina/processed/splits/val.csv` | **366** | `experiments/retina/calibration/v004_temperature_scaling/temperature_scaling.pt` ($T=1.6218$) |
| **Foot ($F$)** | 4-Class Wagner Ulcer Grade | `datasets/foot/processed/splits/val.csv` | **1,006** | `experiments/foot/calibration/vector_scaling/vector_scaling.pt` (`FootVectorScaler`) |
| **Clinical ($C$)** | Binary 30-Day Readmission | `datasets/clinical/processed/splits/val.csv` | **14,911** | `IsotonicCalibrator` (fitted on CatBoost validation margin) |

---

## 2. Modality Pipeline Audit

### 2.1 Retina Pipeline Evidence
- **Architecture**: EfficientNet-B3 with spatial resolution $512 \times 512$.
- **Validation Logits**: Loaded from `validation_logits.npy` ($N=366, K=5$).
- **Post-Calibration Softmax**: Scaled via optimal learned validation temperature $T = 1.6218$:
  $$z'_k = \frac{z_k}{1.6218}, \quad p_k = \frac{e^{z'_k}}{\sum_j e^{z'_j}}$$

### 2.2 Diabetic Foot Ulcer Pipeline Evidence
- **Architecture**: EfficientNet-B3 trained with focal loss and source-image grouping.
- **Validation Logits**: Loaded from `validation_logits.npy` ($N=1,006, K=4$).
- **Post-Calibration Softmax**: Scaled via optimal learned validation Vector Scaling parameters:
  $$z'_k = w_k z_k + b_k, \quad p_k = \frac{e^{z'_k}}{\sum_j e^{z'_j}}$$

### 2.3 Structured Clinical EHR Pipeline Evidence
- **Architecture**: Tuned CatBoost Classifier ($D=119$ features, depth=4, lr=0.1383, iter=350).
- **Validation Predictions**: Evaluated on $N=14,911$ patient encounters across 130 hospitals.
- **Post-Calibration Probability**: Mapped via isotonic regression strictly fitted on validation probabilities:
  $$p_{\text{cal}} = f_{\text{iso}}(p_{\text{raw}})$$
