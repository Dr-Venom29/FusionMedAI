# Step 5: Benchmark Summary Report

## Objective
Identify the optimal model architecture for the Retinal Module through a strictly controlled benchmarking protocol.

## Evaluated Models
1. EfficientNet-B0 (CNN)
2. EfficientNet-B3 (CNN)
3. ConvNeXt-Tiny (CNN)
4. Swin-Tiny (Hierarchical Transformer)
5. ViT-B/16 (Vision Transformer)

## Experimental Protocol
* **Dataset**: APTOS 2019
* **Image Size**: 224x224
* **Optimizer**: AdamW
* **Scheduler**: CosineAnnealingLR
* **Epochs**: 50
* **Seed**: 42

## Results Comparison

| Model | Accuracy | Balanced Acc. | Macro F1 | QWK | ROC-AUC | Params | Peak VRAM | Latency | Throughput |
|---|---|---|---|---|---|---|---|---|---|
| **EfficientNet-B3** | **84.20%** | 67.22% | 0.6813 | **0.9233** | 0.9457 | 10.70M | 2.81 GB | 12.64 ms | 79.1 img/s |
| ConvNeXt-Tiny | 81.20% | **72.05%** | **0.6893** | 0.9145 | **0.9587** | 27.82M | 2.30 GB | 5.65 ms | 177.0 img/s |
| EfficientNet-B0 | 79.29% | 67.68% | 0.6505 | 0.9101 | 0.9353 | 4.01M | 1.50 GB | 8.08 ms | 123.7 img/s |
| Swin-Tiny | 78.75% | 66.35% | 0.6406 | 0.8973 | 0.9516 | 27.52M | 2.57 GB | 12.89 ms | 77.6 img/s |
| ViT-B/16 | 77.38% | 58.01% | 0.5804 | 0.8656 | 0.9225 | 85.80M | 3.41 GB | 15.16 ms | 66.0 img/s |

## Key Findings
* **Highest Accuracy**: EfficientNet-B3
* **Highest QWK**: EfficientNet-B3
* **Highest ROC-AUC**: ConvNeXt-Tiny
* **Fastest Model**: ConvNeXt-Tiny
* **Smallest Model**: EfficientNet-B0
* **Selected Backbone**: EfficientNet-B3

## Conclusion
**EfficientNet-B3** is officially selected as the backbone for the Retinal Module, balancing superior diagnostic capability with reasonable computational overhead.
