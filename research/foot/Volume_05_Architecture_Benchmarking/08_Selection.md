# 08 Selection — Final Backbone Decision

## 10.5.16 Model Selection Decision Matrix

| Metric | ResNet-50 (Baseline Reference) | EfficientNet-B0 (Lightweight Alt) | EfficientNet-B3 (Primary Selected) | Delta ($\Delta$ vs Baseline) | Status / Selection |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Test Macro F1** | `0.6339` | `0.6672` | **`0.6683`** | **+0.0344** | **SELECTED** (Highest Macro F1) |
| **Balanced Accuracy** | `0.6391` | `0.6656` | **`0.6672`** | **+0.0281** | **PASS** |
| **Macro ROC-AUC** | `0.8423` | `0.8431` | **`0.8685`** | **+0.0262** | **PASS** |
| **Grade 2 F1** | — | `0.6585` | **`0.7216`** | — | **PASS** (Strongest Grade 2 recall) |
| **Parameters** | 23.51 M | **4.01 M** | 10.70 M | -12.81 M | **PASS** ($2.2\times$ smaller than baseline) |
| **Batch Latency** | — | **37.89 ms** | 70.39 ms | — | **PASS** |

---

## 10.5.17 Selection Qualification & Statistical Interpretation

EfficientNet-B3 achieved the highest Test Macro F1 (`0.6683`) and Macro ROC-AUC (`0.8685`) among the evaluated candidate architectures and was therefore selected as the primary Foot Ulcer backbone. Its Macro F1 advantage over EfficientNet-B0 (+0.0011) was small, with overlapping 95% bootstrap confidence intervals (`B0: [0.6378, 0.6961]`, `B3: [0.6396, 0.6961]`). Therefore, the result is interpreted as a benchmark-based selection under pre-defined predictive-performance criteria rather than evidence of statistically significant superiority over EfficientNet-B0.

---

## 10.5.18 Final Decision & Model Freeze Manifest

- **Primary Foot Ulcer Backbone**: `EfficientNet-B3`
- **Lightweight Efficiency Alternative**: `EfficientNet-B0`
- **Baseline Reference**: `ResNet-50`
- **Candidate Evidence (Non-Selected)**: `ConvNeXt-Tiny`, `Swin-Tiny`, `ViT-B/16`
- **Freeze Manifest Path**: `experiments/foot/final_model/model_selection.json`
- **Selected Checkpoint Path**: `experiments/foot/architecture_benchmark/efficientnet_b3/checkpoints/best_model.pt`

---

## 10.5.22 Phase Acceptance Criteria

| Criterion | Status |
|---|---|
| Frozen dataset split used | PASS |
| Source-group separation maintained | PASS |
| Exact duplicate leakage absent | PASS |
| Same evaluation protocol | PASS |
| All candidate architectures trained | PASS |
| Test evaluation completed | PASS |
| Per-class metrics generated | PASS |
| Confusion matrices generated | PASS |
| Computational profiling completed | PASS |
| Bootstrap confidence intervals generated | PASS |
| Benchmark artifacts isolated | PASS |
| Primary architecture selected | PASS |

### Phase 10.5 Status
**PASS — Architecture Benchmarking Complete**
- **Selected Primary Backbone**: EfficientNet-B3
- **Lightweight Alternative**: EfficientNet-B0
- **Next Phase**: Phase 10.6 — Final Foot Model
