# Architecture Benchmarking

The fifth phase of FusionMedAI evaluated five distinct vision architectures under a frozen, strictly controlled protocol to select the optimal Retina Module backbone.

## Benchmark Protocol
* **Frozen Settings**: Epochs=50, Patience=10, Batch=32, Image Size=224, Optimizer=AdamW, Scheduler=CosineAnnealingLR, Seed=42
* **Loss Function**: Weighted Cross-Entropy (calculated dynamically)
* **Dataset Splitting**: Identical stratified 70/15/15 splits loaded from Step 2

## Hardware & Profiling
* **Environment**: PyTorch AMP (Automatic Mixed Precision) utilized on CUDA.
* **Tracked Metrics**: FLOPs, MACs, parameter size, peak VRAM, and batch throughput measured programmatically per architecture.

## Final Benchmark Comparison

| Model | Accuracy | Balanced Acc. | Macro F1 | QWK | ROC-AUC | Params | Peak VRAM | Latency | Throughput |
|---|---|---|---|---|---|---|---|---|---|
| **EfficientNet-B3** | **84.20%** | 67.22% | 0.6813 | **0.9233** | 0.9457 | 10.70M | 2.81 GB | 12.64 ms | 79.1 img/s |
| ConvNeXt-Tiny | 81.20% | **72.05%** | **0.6893** | 0.9145 | **0.9587** | 27.82M | 2.30 GB | 5.65 ms | 177.0 img/s |
| EfficientNet-B0 | 79.29% | 67.68% | 0.6505 | 0.9101 | 0.9353 | 4.01M | 1.50 GB | 8.08 ms | 123.7 img/s |
| Swin-Tiny | 78.75% | 66.35% | 0.6406 | 0.8973 | 0.9516 | 27.52M | 2.57 GB | 12.89 ms | 77.6 img/s |
| ViT-B/16 | 77.38% | 58.01% | 0.5804 | 0.8656 | 0.9225 | 85.80M | 3.41 GB | 15.16 ms | 66.0 img/s |

## Selected Backbone

**EfficientNet-B3**

## Results & Discussion

In this benchmark, EfficientNet-B3 achieved the highest overall diagnostic performance, while ConvNeXt-Tiny offered the best efficiency-performance tradeoff. Under the current experimental protocol and dataset, transformer-based architectures did not outperform the CNN-based alternatives.
