# Experiments Documentation

## Overview

FusionMedAI organizes and records experiments by modality and evaluation stage. Configurations, metrics, predictions, checkpoints, logs, and research artifacts are stored in isolated directories to ensure complete auditability and reproducibility.

---

## Experiment Directory Structure

```directory
experiments/
├── retina/
│   ├── architecture_benchmark/
│   ├── calibration/
│   ├── explainability/
│   └── final_model/
├── foot/
│   ├── baseline/
│   ├── architecture_benchmark/
│   ├── explainability/
│   ├── calibration/
│   ├── uncertainty/
│   └── final_model/
└── clinical/
    ├── benchmarking/
    ├── catboost_hpo/
    ├── explainability/
    ├── calibration/
    ├── uncertainty/
    ├── robustness/
    └── integration/
```

This structure maintains strict isolation between experimental runs and prevents configuration or checkpoint collision.

---

## Experiment Tracking and Artifacts

For each completed experiment, the framework records:
- **Configuration**: Model architecture, hyperparameters, random seed, optimizer, loss function, and scheduler.
- **Evaluation Tables**: Partitioned validation and held-out test performance tables.
- **Visual Artifacts**: Reliability diagrams, ROC/PR curves, Grad-CAM overlays, TreeSHAP summary plots, and risk-coverage curves.
- **Cryptographic Manifests**: SHA-256 digests linking code, model weights, evaluation tables, and figures.

---

## 1. Retina Module Experiments — Completed

### Architecture Benchmarking
- Evaluated 5 vision backbones: EfficientNet-B0, EfficientNet-B3, ConvNeXt-Tiny, Swin-Tiny, and ViT-B/16.
- **Selected Backbone**: EfficientNet-B3 (Accuracy: $84.20\%$, QWK: $0.9233$, Macro F1: $0.6813$, ROC-AUC: $0.9457$).

### Probability Calibration
- Post-hoc Temperature Scaling on the validation partition ($N=366$).
- Reduced test Expected Calibration Error (ECE) to $0.0241$ while strictly preserving classification ranking.

### Prediction Uncertainty Estimation
- Monte Carlo Dropout with $N^*=25$ stochastic forward passes on held-out test scans.
- Quantified predictive entropy, mutual information, and predictive variance ($\text{Error Detection AUROC} = 0.8443$).

### Post-Hoc Explainability
- Spatial Grad-CAM localization on convolutional feature activations.

---

## 2. Diabetic Foot Ulcer Module Experiments — Completed

### Baseline Framework
- ResNet-50 baseline on canonical source-group splits (Macro F1: $0.6339$, Balanced Accuracy: $0.6391$, Macro ROC-AUC: $0.8423$).

### Architecture Benchmarking
- Evaluated 6 candidate backbones on the 4-class Wagner classification task.
- **Selected Backbone**: EfficientNet-B3 (Macro F1: $0.6683$, Balanced Accuracy: $0.6672$, Macro ROC-AUC: $0.8685$).

### Post-Hoc Explainability & Sanity Checking
- Spatial Grad-CAM at `backbone.features[8]` highlighting wound bed and margin boundaries.
- Model parameter randomization sanity check confirmed sensitivity ($\rho = 0.0000$).

### Probability Calibration
- Evaluated Temperature Scaling and Vector Scaling.
- **Selected Method**: Vector Scaling (Test NLL: $0.8749$, Test ECE: $0.0313$, $+26.18\%$ relative ECE reduction).

### Uncertainty Estimation & Selective Prediction
- 10-pass MC Dropout quantified predictive entropy ($\text{AUROC} = 0.7291$, $\text{AUPRC} = 0.5391$).
- Risk-coverage analysis demonstrated monotonic test error rate reduction from $32.31\%$ to $11.20\%$ at $50\%$ coverage.

---

## 3. Clinical Module Experiments — Completed

The Clinical Module evaluates 30-day readmission risk on the UCI Diabetes dataset ($N=101,766$) using a frozen 119-dimensional representation and patient-level canonical partitioning.

### Model Selection & Validation HPO (C5)
- Benchmarked 7 tabular architectures: CatBoost, XGBoost, LightGBM, Logistic Regression (L2, ElasticNet), Random Forest, and TabNet.
- Bounded 15-trial validation-only Optuna search for CatBoost (`depth=4`, `learning_rate=0.1383`, `iterations=350`, `l2_leaf_reg=2.911`, `subsample=0.655`, `random_seed=42`).
- **Tuned Result**: Test ROC-AUC $0.6504$, Test PR-AUC $0.2063$, Test Brier $0.0952$.

### Model Explainability (C6)
- Exact TreeSHAP decomposition on the frozen CatBoost model across the locked test partition ($N=14,913, D=119$).
- Ranked feature attributions: `number_inpatient` ($22.43\%$ share), `age_ordinal` ($7.86\%$), `time_in_hospital` ($6.41\%$).
- Validation vs test rank correlation confirmed stability ($\rho = 0.9994$, $100\%$ Top-20 overlap).

### Probability Calibration & Decision Analysis (C7)
- Evaluated Raw, Platt Scaling, Beta Calibration, and Isotonic Regression.
- **Protocol Selection**: Isotonic Regression selected on validation NLL ($0.3420$). Out-of-sample Beta achieved parametric slope $0.9720$.
- Decision Curve Analysis confirmed positive clinical net benefit across $\theta \in [0.05, 0.25]$.

### Prediction Uncertainty Estimation (C8)
- 50-member Bootstrap CatBoost Ensemble ($N_{\text{train}}=69,519$).
- Evaluated predictive dispersion ($\mu_{\sigma} = 0.0219$, median $\sigma_p = 0.0162$).
- Uncertainty discriminated misclassified cases ($\sigma_p = 0.0357$) from correct predictions ($\sigma_p = 0.0195$; Error AUROC: $0.7116$, AUPRC: $0.3256$).
- Rejection at $80\%$ coverage reduced error from $14.83\%$ to $10.23\%$. Stratified cohort into 6 ambiguity decision tiers.

### Robustness & Distribution-Shift Auditing (C9)
- Evaluated 11 perturbation scenarios: MCAR missingness (+10%, +25%, +50%), targeted feature masks, demographic strata, and temporal eras (1999–2003 vs 2004–2008).
- Confirmed uncertainty inflation response under random missingness ($\sigma_p = 0.0219 \to 0.0491$).
- Identified prior-inpatient uncertainty blind spot (ROC-AUC drops to $0.5795$ while $\sigma_p$ falls to $0.0150$).

### Clinical Integration & End-to-End Validation (C10)
- Assembled frozen C5–C9 components into `ClinicalInferenceService`.
- Standardized `ClinicalOutput` schema containing prediction, calibrated probability, predictive intervals, TreeSHAP attributions, shift alerts, and provenance.
- Verified exact TreeSHAP margin additivity ($\text{error} = 8.88 \times 10^{-16} \le 10^{-6}$).
- Evaluated 14,913 locked test encounters in $4.46\text{ seconds}$ ($3,345.7\text{ encounters/sec}$ local CPU batch throughput).

---

## Evaluation Criteria

### Image Modules (Retina & Foot Ulcer)
- **Discrimination**: Macro F1-score, Top-1 Accuracy, Balanced Accuracy, Quadratic Weighted Kappa (QWK), Macro ROC-AUC.
- **Calibration**: Multi-class Expected Calibration Error (ECE), Negative Log-Likelihood (NLL), Brier Score.
- **Uncertainty**: Error Detection AUROC / AUPRC, Predictive Entropy, Predictive Variance, Mutual Information, Risk-Coverage AURC.
- **Computational Cost**: Parameters, FLOPs, Peak VRAM, Latency (ms), Throughput (img/s).

### Clinical Tabular Module
- **Discrimination**: Binary ROC-AUC, PR-AUC, Sensitivity, Specificity.
- **Probability Quality**: Test Log Loss, Brier Score, ECE, Parametric Calibration Slope, Decision Curve Net Benefit.
- **Interpretability**: TreeSHAP Mean |SHAP|, Feature Group Shares, Rank Correlation ($\rho$).
- **Uncertainty & Selectivity**: Bootstrap Dispersion ($\sigma_p$), 95% Predictive Intervals, Error AUROC/AUPRC, Risk-Coverage AURC, Excess AURC.
- **Robustness**: $\Delta\text{ROC-AUC}$, $\Delta\text{Slope}$, Uncertainty Inflation Ratio, Silent Failure Rates.
- **Integration**: Schema conformance, physiological bound rejection, exact additivity, local CPU batch throughput.

---

## Future Experimental Directions

*These directions outline prospective research extensions and are not part of the currently reported baseline results.*

### Hyperparameters & Optimization
- Adaptive learning rate schedules and gradient accumulation for large image batch regimes.
- Tabular tree pruning depth vs calibration slope trade-off studies.

### Loss Formulations
- Class-balanced focal loss and label smoothing cross-entropy for extreme long-tail distributions.
- Direct calibration-regularized training objectives.

---

## Next Research Stage: ACARA-U Multimodal Fusion

The next major research stage is **ACARA-U Multimodal Fusion**.

ACARA-U will consume the independently validated outputs of the Retina, Foot Ulcer, and Clinical modules rather than merging their underlying datasets:
- Evaluates individual-modality baselines vs pairwise and three-way decision-level fusion.
- Explores reliability-weighted decision aggregation driven by modality predictive uncertainty ($\sigma_p$, predictive entropy).
- Audits multimodal calibration, partial-input missing-modality tolerance, and stress-tested distribution shifts.
