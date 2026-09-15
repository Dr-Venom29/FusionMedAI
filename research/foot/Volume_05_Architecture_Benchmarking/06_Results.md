# 06 Results — Phase 10.5 Benchmark Summary

## 10.5.9 Primary Benchmark Ranking

| Rank | Architecture | Test Macro F1 | Balanced Accuracy | Macro ROC-AUC | Parameters (M) | Latency (ms) |
|---:|---|---:|---:|---:|---:|---:|
| 🥇 | **EfficientNet-B3** | **0.6683** | **0.6672** | **0.8685** | 10.70 M | 70.39 ms |
| 🥈 | **EfficientNet-B0** | **0.6672** | **0.6656** | **0.8431** | 4.01 M | 37.89 ms |
| 🥉 | **ConvNeXt-Tiny** | **0.6566** | **0.6562** | **0.8613** | 27.82 M | 109.61 ms |
| 4 | **ResNet-50 (Baseline Reference)** | **0.6339** | **0.6391** | **0.8423** | 23.51 M | — |
| 5 | **Swin-Tiny** | **0.6266** | **0.6262** | **0.8269** | 27.52 M | 129.09 ms |
| 6 | **ViT-B/16** | **0.5788** | **0.5878** | **0.8364** | 85.80 M | 310.28 ms |

---

## 10.5.10 Complete Evaluation Results

| Model | Test Loss | Accuracy | Bal. Acc. | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **ResNet-50** | 1.0548 | 0.6372 | 0.6391 | — | — | 0.6339 | 0.6319 | 0.8423 |
| **EfficientNet-B0** | 1.1251 | 0.6670 | 0.6656 | 0.6698 | 0.6656 | **0.6672** | 0.6676 | 0.8431 |
| **EfficientNet-B3** | 0.8779 | 0.6690 | 0.6672 | 0.6729 | 0.6672 | **0.6683** | 0.6682 | **0.8685** |
| **ConvNeXt-Tiny** | 1.0271 | 0.6521 | 0.6562 | 0.6653 | 0.6562 | 0.6566 | 0.6541 | 0.8613 |
| **Swin-Tiny** | 1.5243 | 0.6252 | 0.6262 | 0.6338 | 0.6262 | 0.6266 | 0.6249 | 0.8269 |
| **ViT-B/16** | 1.2080 | 0.5885 | 0.5878 | 0.6079 | 0.5878 | 0.5788 | 0.5773 | 0.8364 |

---

## 10.5.11 Bootstrap Confidence Intervals

Image-level 95% bootstrap confidence intervals ($B=1,000$ resamples, seed `42`).

| Model | Test Macro F1 | 95% Bootstrap CI | Balanced Accuracy | 95% Bootstrap CI |
|---|---:|---|---:|---|
| **EfficientNet-B0** | 0.6672 | [0.6378, 0.6961] | 0.6656 | [0.6360, 0.6948] |
| **EfficientNet-B3** | **0.6683** | [0.6396, 0.6961] | **0.6672** | [0.6380, 0.6953] |
| **ConvNeXt-Tiny** | 0.6566 | [0.6266, 0.6846] | 0.6562 | [0.6261, 0.6844] |
| **Swin-Tiny** | 0.6266 | [0.5943, 0.6557] | 0.6262 | [0.5940, 0.6552] |
| **ViT-B/16** | 0.5788 | [0.5477, 0.6069] | 0.5878 | [0.5568, 0.6159] |

---

## 10.5.12 Class-Wise Performance Breakdown

| Class | EfficientNet-B0 | EfficientNet-B3 | ConvNeXt-Tiny | Swin-Tiny | ViT-B/16 |
|:--- |---:|---:|---:|---:|---:|
| **Grade 1 F1** | 0.6276 | 0.5818 | 0.6328 | **0.6540** | 0.6151 |
| **Grade 2 F1** | 0.6585 | **0.7216** | 0.6391 | 0.5389 | 0.4309 |
| **Grade 3 F1** | **0.6712** | 0.6456 | 0.5860 | 0.5905 | 0.5562 |
| **Grade 4 F1** | 0.7115 | 0.7243 | **0.7685** | 0.7231 | 0.7129 |
