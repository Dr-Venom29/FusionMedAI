# Chapter 6: Performance vs. Efficiency and Model Selection

## 6.1 Computational Requirements
Deploying clinical AI requires balancing diagnostic power with inference constraints.

- **Latency & Throughput**: ConvNeXt-Tiny achieved the highest throughput (177.0 img/s) and lowest latency (5.65 ms). EfficientNet-B3 maintained highly respectable speeds (79.1 img/s), easily sufficient for real-time or batch clinical inference.
- **VRAM Constraints**: ViT-B/16 demanded the highest peak VRAM (3.41 GB), while EfficientNet-B0 was extremely light (1.50 GB). EfficientNet-B3 sat comfortably in the middle (2.81 GB), allowing it to easily fit on commodity clinical hardware.
- **Parameter Efficiency**: EfficientNet-B3 achieved the best accuracy using only 10.70M parameters, highlighting the power of compound scaling compared to Swin-Tiny (27.52M) and ViT-B/16 (85.80M).

## 6.2 Deployment Recommendations

Based on the benchmark results, the models are suited for different deployment environments:

| Deployment Target | Recommended Model |
| :--- | :--- |
| Edge device | EfficientNet-B0 |
| General clinical deployment | EfficientNet-B3 |
| High-throughput server | ConvNeXt-Tiny |
| Research only | ViT-B/16 |

## 6.3 Selection Rationale
Based strictly on the benchmark results, **EfficientNet-B3** has been selected as the official Retinal Module backbone for FusionMedAI.

## 6.4 Justification
1. **Clinical Superiority**: It achieved the highest absolute Accuracy (84.20%) and Quadratic Weighted Kappa (0.9233), which is the primary metric for grading retinopathy severity.
2. **Efficiency**: At only 10.70M parameters, it heavily outperforms massive models like ViT-B/16, validating that inductive bias is still highly valuable in data-constrained medical imaging.
3. **Deployment Feasibility**: A latency of 12.64 ms and peak VRAM of 2.81 GB means the model can be deployed locally on edge devices or standard clinic PCs without requiring high-end datacenter GPUs.

While ConvNeXt-Tiny showed strong ROC-AUC, EfficientNet-B3's superior accuracy and QWK make it the safer, more robust choice for final grading.
