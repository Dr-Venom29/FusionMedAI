# 03 Model Configurations — Candidate Registries

## 10.5.7 Candidate Architecture Registries

All candidate architectures were instantiated via central factory `src/foot/models/factory.py` using `MODEL_REGISTRY` with uniform classifier head dropout ($p = 0.2$).

### 1. ResNet-50 (Phase 10.4 Baseline Reference)
- Standard residual convolutional architecture serving as the baseline reference point.
- **Parameters**: 23.51 M | **Model Size**: 89.69 MB

### 2. EfficientNet-B0
- Lightweight compound-scaled convolutional architecture evaluated as an efficient baseline alternative.
- **Parameters**: 4.01 M | **Model Size**: 15.31 MB

### 3. EfficientNet-B3
- Higher-capacity EfficientNet architecture evaluated as the primary candidate for the final Foot backbone.
- **Parameters**: 10.70 M | **Model Size**: 40.83 MB

### 4. ConvNeXt-Tiny
- Modern convolutional architecture incorporating depthwise separable convolutions and inverted bottleneck layers.
- **Parameters**: 27.82 M | **Model Size**: 106.14 MB

### 5. Swin-Tiny
- Hierarchical vision transformer architecture utilizing shifted-window local self-attention.
- **Parameters**: 27.52 M | **Model Size**: 104.99 MB

### 6. ViT-B/16
- Large vision transformer architecture utilizing global patch self-attention ($16 \times 16$ patches).
- **Parameters**: 85.80 M | **Model Size**: 327.31 MB

---

## 10.5.8 Model Computational & Hardware Profiling

All candidate vision architectures were evaluated using fixed $224 \times 224 \times 3$ RGB inputs to prevent input resolution from acting as a confounding variable.

| Architecture | Parameters (M) | Model Size (MB) | Batch Latency (ms) | Throughput (fps) | Profiling Device |
|:--- |---:|---:|---:|---:|:---:|
| **EfficientNet-B0** | 4.01 M | 15.31 MB | 37.89 ms | 844.55 fps | CUDA |
| **EfficientNet-B3** | 10.70 M | 40.83 MB | 70.39 ms | 454.61 fps | CUDA |
| **ConvNeXt-Tiny** | 27.82 M | 106.14 MB | 109.61 ms | 291.94 fps | CUDA |
| **Swin-Tiny** | 27.52 M | 104.99 MB | 129.09 ms | 247.89 fps | CUDA |
| **ViT-B/16** | 85.80 M | 327.31 MB | 310.28 ms | 103.13 fps | CUDA |

*Latency and throughput measured programmatically via `profile_model()` using `torch.inference_mode()`, batch size 32, $224 \times 224$ resolution, and `torch.cuda.synchronize()` CUDA event timing.*
