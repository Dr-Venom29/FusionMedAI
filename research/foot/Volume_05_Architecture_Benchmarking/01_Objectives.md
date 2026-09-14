# 01 Objectives — Foot Architecture Benchmarking

## Phase 10.5 Objectives

1. **Evaluate Candidate Families**: Benchmark 5 distinct vision backbone architectures (`EfficientNet-B0`, `EfficientNet-B3`, `ConvNeXt-Tiny`, `Swin-Tiny`, `ViT-B/16`) under a controlled experimental protocol.
2. **Methodological Continuity**: Maintain exact protocol alignment with the Retina Module architecture benchmarking suite.
3. **Parameter & Hardware Profiling**: Programmatically inspect total parameters, trainable parameters, memory size, and inference throughput.
4. **Outperform Reference Baseline**: Evaluate whether candidate architectures outperform the frozen Phase 10.4 ResNet-50 baseline (**Macro F1: 0.6339**).
5. **Backbone Selection**: Select the final Foot Ulcer module backbone based on classification metrics, parameter efficiency, and latency.
