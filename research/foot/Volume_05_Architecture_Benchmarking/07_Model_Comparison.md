# 07 Model Comparison — Tradeoff Analysis

## Comparative Analysis Criteria

Candidate architectures evaluated in Phase 10.5 are compared across three core dimensions:
1. **Classification Performance**: Test Macro F1 (primary metric), Accuracy, Balanced Accuracy, Macro ROC-AUC, and Class-wise Recall (focusing on Grade 2 ↔ Grade 3 boundary).
2. **Computational Efficiency**: Total Parameter Count (M) and Model Weight Size (MB).
3. **Inference Latency & Throughput**: Average batch latency (ms for batch size 32) and throughput (frames per second).

---

## Baseline Reference vs. Candidate Architectures

The Phase 10.4 baseline model (ResNet-50) serves as the frozen baseline reference for all candidate architecture comparisons.

### Reference Baseline Performance (Phase 10.4)
- **Architecture**: ResNet-50 (ImageNet Pre-trained)
- **Macro F1**: `0.6339`
- **Accuracy**: `0.6372`
- **Balanced Accuracy**: `0.6391`
- **Macro ROC-AUC**: `0.8423`
- **Parameter Count**: `23.51 M`
- **Model Size**: `89.69 MB`

---

## Model Profiling & Performance Comparison Table

| Architecture | Macro F1 | Accuracy | Balanced Acc. | Macro ROC-AUC | Params (M) | Size (MB) | Latency (ms) | Throughput (fps) | Relative to ResNet-50 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-50 (Baseline)** | **0.6339** | **0.6372** | **0.6391** | **0.8423** | **23.51** | **89.69** | — | — | **Reference** |
| EfficientNet-B0 | — | — | — | — | 4.01 | 15.31 | 55.49 | 288.35 | Pending Benchmark |
| EfficientNet-B3 | — | — | — | — | 10.70 | 40.83 | 114.73 | 139.46 | Pending Benchmark |
| ConvNeXt-Tiny | — | — | — | — | 27.82 | 106.14 | 93.30 | 171.49 | Pending Benchmark |
| Swin-Tiny | — | — | — | — | 27.52 | 104.99 | 102.50 | 156.10 | Pending Benchmark |
| ViT-B/16 | — | — | — | — | 85.80 | 327.31 | 240.23 | 66.60 | Pending Benchmark |

*Note: Profiling parameters, model size, and CPU latency measured programmatically via `profile_model()`. Benchmark metrics (Macro F1, Accuracy, etc.) will be populated upon benchmark execution.*

---

## Statistical Significance & Bootstrap Comparison

To prevent selecting a model based on nominal differences, all candidate comparisons against the ResNet-50 baseline will incorporate 95% Bootstrap Confidence Intervals (B=1,000 iterations):
- **$\Delta \text{Macro F1} = \text{Macro F1}_{\text{candidate}} - \text{Macro F1}_{\text{ResNet-50}}$**
- A candidate is declared statistically superior to the baseline **only if** the 95% CI of $\Delta \text{Macro F1}$ excludes zero (lower bound > 0).
