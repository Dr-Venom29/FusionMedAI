# Chapter 5: Benchmark Results

## 5.1 Final Comparison Table

| Model | Accuracy | Balanced Acc. | Macro F1 | QWK | ROC-AUC | Params | Peak VRAM | Latency | Throughput |
|---|---|---|---|---|---|---|---|---|---|
| EfficientNet-B3 | 84.20% | 67.22% | 0.6813 | 0.9233 | 0.9457 | 10.70M | 2.81 GB | 12.64 ms | 79.1 img/s |
| ConvNeXt-Tiny | 81.20% | 72.05% | 0.6893 | 0.9145 | 0.9587 | 27.82M | 2.30 GB | 5.65 ms | 177.0 img/s |
| EfficientNet-B0 | 79.29% | 67.68% | 0.6505 | 0.9101 | 0.9353 | 4.01M | 1.50 GB | 8.08 ms | 123.7 img/s |
| Swin-Tiny | 78.75% | 66.35% | 0.6406 | 0.8973 | 0.9516 | 27.52M | 2.57 GB | 12.89 ms | 77.6 img/s |
| ViT-B/16 | 77.38% | 58.01% | 0.5804 | 0.8656 | 0.9225 | 85.80M | 3.41 GB | 15.16 ms | 66.0 img/s |

## 5.2 Winner by Metric

| Metric | Best Model |
| :--- | :--- |
| Accuracy | EfficientNet-B3 |
| QWK | EfficientNet-B3 |
| Balanced Accuracy | ConvNeXt-Tiny |
| Macro F1 | ConvNeXt-Tiny |
| ROC-AUC | ConvNeXt-Tiny |
| Latency | ConvNeXt-Tiny |
| Throughput | ConvNeXt-Tiny |
| Parameters | EfficientNet-B0 |
| VRAM | EfficientNet-B0 |

## 5.3 Discussion of Findings
- **Highest Accuracy & QWK**: EfficientNet-B3 achieved the highest absolute accuracy (84.20%) and QWK (0.9233), making it the strongest clinical classifier among the candidates.
- **Highest ROC-AUC**: ConvNeXt-Tiny showed the highest ROC-AUC (0.9587) and Balanced Accuracy (72.05%), indicating exceptional separability.
- **Smallest Footprint**: EfficientNet-B0 required the least VRAM (1.50 GB) and parameters (4.01M), but sacrificed diagnostic accuracy.
- **Transformer Performance**: ViT-B/16 struggled on this dataset size, yielding the worst diagnostic metrics across the board despite having the largest parameter count (85.80M). Transformers require significantly larger training regimes to outperform CNNs without strong inductive biases.
