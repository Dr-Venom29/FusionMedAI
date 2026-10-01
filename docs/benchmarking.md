# Architecture Benchmarking

## Overview

FusionMedAI conducts controlled architecture benchmarks for each input modality. Candidate architectures are evaluated under identical data partitions, loss functions, and evaluation protocols to select the primary modality backbone.

---

## 1. Retina Module Backbone Benchmark

The Retina backbone was selected through a controlled comparative evaluation of CNN and transformer-based vision architectures on the APTOS 2019 dataset.

### Benchmark Protocol
- **Training Settings**: Epochs = 50, Patience = 10, Batch Size = 32, Image Size = 224 × 224, Optimizer = AdamW, Scheduler = CosineAnnealingLR, Seed = 42
- **Loss Function**: Dynamically weighted Cross-Entropy
- **Dataset Partitioning**: Stratified 80/10/10 split ($N_{\text{test}} = 367$)
- **Evaluation Hardware**: PyTorch Automatic Mixed Precision (AMP) on CUDA

### Benchmark Comparison

| Rank | Model Architecture | Accuracy | Balanced Acc. | Macro F1 | Quadratic Weighted Kappa (QWK) | Test ROC-AUC | Parameters | Peak VRAM | Latency (GPU) | Throughput |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **EfficientNet-B3** (Selected) | **84.20%** | 67.22% | 0.6813 | **0.9233** | 0.9457 | 10.70M | 2.81 GB | 12.64 ms | 79.1 img/s |
| 2 | **ConvNeXt-Tiny** | 81.20% | **72.05%** | **0.6893** | 0.9145 | **0.9587** | 27.82M | 2.30 GB | **5.65 ms** | **177.0 img/s** |
| 3 | **EfficientNet-B0** | 79.29% | 67.68% | 0.6505 | 0.9101 | 0.9353 | **4.01M** | **1.50 GB** | 8.08 ms | 123.7 img/s |
| 4 | **Swin-Tiny** | 78.75% | 66.35% | 0.6406 | 0.8973 | 0.9516 | 27.52M | 2.57 GB | 12.89 ms | 77.6 img/s |
| 5 | **ViT-B/16** | 77.38% | 58.01% | 0.5804 | 0.8656 | 0.9225 | 85.80M | 3.41 GB | 15.16 ms | 66.0 img/s |

### Selection Rationale & Scientific Interpretation
**EfficientNet-B3** was selected as the Retina Module backbone according to the predefined backbone-selection protocol.

It achieved the highest overall Accuracy ($84.20\%$) and Quadratic Weighted Kappa ($0.9233$) among evaluated architectures. ConvNeXt-Tiny achieved higher Balanced Accuracy, Macro F1, and ROC-AUC, alongside lower latency and higher throughput. The selection therefore reflects the predefined evaluation protocol rather than uniform dominance across every single reported metric.

---

## 2. Diabetic Foot Ulcer Backbone Benchmark

The Foot Ulcer backbone was selected through a 6-architecture benchmark on the ADPM V3.3 dataset ($10,062$ audited images grouped into $1,770$ canonical source-image clusters to prevent patient leakage).

### Benchmark Protocol
- **Training Settings**: Epochs = 50, Early Stopping Patience = 10, Batch Size = 32, Image Size = 224 × 224, Optimizer = AdamW, Seed = 42
- **Data Partitions**: Canonical group-stratified splits (Train: 8,038, Val: 1,006, Held-out Test: 1,006)
- **Primary Metric**: Held-out Test Macro F1 across 4 Wagner grades

### Benchmark Comparison

| Rank | Model Architecture | Macro F1 | Balanced Accuracy | Macro ROC-AUC | Parameters | Latency (GPU, T4) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | **EfficientNet-B3** (Selected) | **0.6683** | **0.6672** | **0.8685** | **10.70M** | **70.39 ms** |
| 2 | **EfficientNet-B0** | 0.6672 | 0.6656 | 0.8431 | 4.01M | 37.89 ms |
| 3 | **ConvNeXt-Tiny** | 0.6566 | 0.6562 | 0.8613 | 27.82M | 109.61 ms |
| 4 | **ResNet-50** (Baseline Reference) | 0.6339 | 0.6391 | 0.8423 | 23.51M | — |
| 5 | **Swin-Tiny** | 0.6266 | 0.6262 | 0.8269 | 27.52M | 129.09 ms |
| 6 | **ViT-B/16** | 0.5788 | 0.5878 | 0.8364 | 85.80M | 310.28 ms |

### Selection Rationale & Scientific Interpretation
**EfficientNet-B3** was selected as the primary Foot Ulcer backbone based on achieving the highest held-out test Macro F1 ($0.6683$) and Balanced Accuracy ($0.6672$). EfficientNet-B0 remains a viable lightweight alternative, scoring within $0.0011$ Macro F1 while requiring fewer parameters.

---

## 3. Structured Clinical Tabular Benchmark

The Clinical tabular architecture was selected through a 7-architecture benchmarking and validation-only hyperparameter optimization (HPO) protocol on the UCI Diabetes dataset ($N=99,343, D=119$).

### Benchmark Protocol
- **Feature Space**: Locked 119-dimensional clinical representation
- **Splits**: Patient-level canonical partitioning ($N_{\text{train}}=69,519, N_{\text{val}}=14,911, N_{\text{test}}=14,913$)
- **Optimization Criterion**: Validation Negative Log-Likelihood (the locked test partition was strictly held out during HPO)

### Benchmark Comparison

| Architecture | Test ROC-AUC | Test PR-AUC | Test Brier | Test ECE | Train Time | Latency / 1k | Serialized Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CatBoost (HPO Tuned)** | **0.6504** | **0.2063** | **0.0952** | 0.0053 | 3.87 s | 2.35 ms | 415.6 KB |
| **CatBoost (Baseline)** | 0.6472 | 0.2038 | 0.0953 | 0.0066 | 3.87 s | 2.35 ms | 415.6 KB |
| **XGBoost** | 0.6467 | 0.2035 | 0.0953 | 0.0051 | 1.20 s | 1.65 ms | 260.2 KB |
| **LightGBM** | 0.6461 | 0.2038 | 0.0953 | **0.0045** | **0.58 s** | 3.18 ms | 318.2 KB |
| **Logistic Regression (L2)** | 0.6446 | 0.1969 | 0.0958 | 0.0080 | 2.09 s | **1.13 ms** | **1.8 KB** |
| **Logistic Regression (ElasticNet)** | 0.6445 | 0.1971 | 0.0958 | 0.0084 | 2.45 s | 1.15 ms | 1.8 KB |
| **Random Forest** | 0.6422 | 0.1991 | 0.0959 | 0.0098 | 2.58 s | 44.83 ms | 5,519.4 KB |
| **TabNet** | 0.6252 | 0.1887 | 0.0962 | 0.0105 | 77.27 s | 19.93 ms | 1,099.7 KB |

### Frozen Configuration (CatBoost HPO)
- **Depth**: `4`
- **Learning Rate**: `0.1383`
- **Iterations**: `350`
- **L2 Leaf Regularization**: `2.911`
- **Subsample Ratio**: `0.655`
- **Random Seed**: `42`

### Selection Rationale & Scientific Interpretation
Gradient-boosted decision tree architectures demonstrated superior tabular discrimination and calibration over linear baselines and deep tabular models (TabNet). Tuned CatBoost achieved the highest test ROC-AUC ($0.6504$) and PR-AUC ($0.2063$). The moderate discrimination score reflects the intrinsic difficulty of retrospective tabular readmission prediction.

---

## Benchmarking Principles

- **Controlled Comparison**: Architectures within each benchmark use identical dataset splits and prescribed training configurations.
- **Metric Consistency**: Classification, calibration, and ranking metrics are computed using standardized evaluation pipelines.
- **Computational Profiling**: Parameter count, latency, throughput, memory usage, and serialized sizes are programmatically measured.
- **Predefined Selection Criteria**: Model selection is strictly based on criteria and constraints registered prior to test evaluation.
- **Reproducibility**: Random seeds, configurations, and benchmark artifacts are cryptographically recorded with each run.
