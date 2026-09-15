# 01 Objectives — Foot Architecture Benchmarking

## 10.5.1 Phase 10.5 Objectives

1. **Evaluate Candidate Vision Families**: Benchmark 5 distinct vision backbone architectures (`EfficientNet-B0`, `EfficientNet-B3`, `ConvNeXt-Tiny`, `Swin-Tiny`, `ViT-B/16`) under a controlled and reproducible experimental protocol for 4-class Wagner diabetic foot ulcer (DFU) classification.
2. **Methodological Continuity**: Maintain exact protocol alignment with the Retina Module architecture benchmarking suite while respecting Foot modality dataset characteristics.
3. **Parameter & Computational Profiling**: Programmatically inspect total parameters (M), model size (MB), average batch latency (ms), and throughput (fps) on CUDA hardware.
4. **Outperform Reference Baseline**: Determine whether candidate vision architectures improve upon the frozen Phase 10.4 ResNet-50 baseline (**Test Macro F1: 0.6339**, **Balanced Accuracy: 0.6391**, **Macro ROC-AUC: 0.8423**).
5. **Backbone Selection & Freezing**: Formally select and freeze the primary Foot Ulcer classification backbone according to the pre-defined predictive-performance metric criteria.

---

## 10.5.2 Research Position

Phase 10.4 established the ResNet-50 baseline reference. Phase 10.5 extends this baseline with controlled architecture benchmarking across 5 candidate architectures:

1. **EfficientNet-B0** (Lightweight convolutional baseline alternative)
2. **EfficientNet-B3** (Higher-capacity compound-scaled CNN candidate)
3. **ConvNeXt-Tiny** (Modern depthwise-conv inverted bottleneck architecture)
4. **Swin-Tiny** (Hierarchical shifted-window vision transformer candidate)
5. **ViT-B/16** (Global self-attention vision transformer candidate)

*Note: ResNet-50 is retained as the Phase 10.4 baseline reference point rather than being retrained as part of the Phase 10.5 candidate set.*
