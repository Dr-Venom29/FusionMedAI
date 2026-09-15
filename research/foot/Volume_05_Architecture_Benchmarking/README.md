# Volume V — Foot Ulcer Architecture Benchmarking

## Phase 10.5 — Architecture Benchmarking

This research volume documents the controlled benchmarking of candidate vision architectures for the **Foot Ulcer (DFU) Wagner 4-Class Classification** modality.

The goal of Phase 10.5 is to evaluate five candidate vision architectures under a frozen, reproducible experimental protocol to select the optimal model backbone relative to the ResNet-50 baseline.

---

## Volume Structure

1. **[01 Objectives](file:///d:/FusionMedAI/research/foot/Volume_05_Architecture_Benchmarking/01_Objectives.md)** — Objectives, research position, and baseline reference position.
2. **[02 Benchmark Protocol](file:///d:/FusionMedAI/research/foot/Volume_05_Architecture_Benchmarking/02_Benchmark_Protocol.md)** — Population, split, class definitions, and frozen parameters.
3. **[03 Model Configurations](file:///d:/FusionMedAI/research/foot/Volume_05_Architecture_Benchmarking/03_Model_Configurations.md)** — Candidate registries and computational hardware profiling.
4. **[04 Training Protocol](file:///d:/FusionMedAI/research/foot/Volume_05_Architecture_Benchmarking/04_Training_Protocol.md)** — Fresh state isolation, GPU memory release, and artifact structure.
5. **[05 Evaluation Protocol](file:///d:/FusionMedAI/research/foot/Volume_05_Architecture_Benchmarking/05_Evaluation_Protocol.md)** — Test-set isolation, evaluation order, CIs, and overfitting analysis.
6. **[06 Results](file:///d:/FusionMedAI/research/foot/Volume_05_Architecture_Benchmarking/06_Results.md)** — Primary ranking, complete evaluation metrics, and bootstrap CIs.
7. **[07 Model Comparison](file:///d:/FusionMedAI/research/foot/Volume_05_Architecture_Benchmarking/07_Model_Comparison.md)** — Tradeoff analysis, boundary confusion, and architecture observations.
8. **[08 Selection](file:///d:/FusionMedAI/research/foot/Volume_05_Architecture_Benchmarking/08_Selection.md)** — Selection decision matrix, statistical qualification, and acceptance status.

---

## Primary Benchmark Ranking & Decision

| Rank | Architecture | Test Macro F1 | Balanced Accuracy | Macro ROC-AUC | Parameters | Latency (ms) | Status |
|---:|---|---:|---:|---:|---:|---:|:--- |
| 🥇 | **EfficientNet-B3** | **0.6683** | **0.6672** | **0.8685** | 10.70 M | 70.39 ms | **SELECTED (Primary Backbone)** |
| 🥈 | **EfficientNet-B0** | **0.6672** | **0.6656** | **0.8431** | 4.01 M | 37.89 ms | **Lightweight Alternative** |
| 🥉 | **ConvNeXt-Tiny** | **0.6566** | **0.6562** | **0.8613** | 27.82 M | 109.61 ms | Benchmark Candidate |
| 4 | **ResNet-50 (Baseline)** | **0.6339** | **0.6391** | **0.8423** | 23.51 M | — | Baseline Reference |
| 5 | **Swin-Tiny** | **0.6266** | **0.6262** | **0.8269** | 27.52 M | 129.09 ms | Underperforming Transformer |
| 6 | **ViT-B/16** | **0.5788** | **0.5878** | **0.8364** | 85.80 M | 310.28 ms | Underperforming Transformer |

- **Phase Status**: **PASS — Complete**
- **Selected Backbone**: `EfficientNet-B3`
- **Next Phase**: Phase 10.6 — Final Foot Model
