# 03 Model Configurations — Candidate Registries

## Controlled Input Resolution Protocol

All candidate vision architectures are evaluated using the fixed $224 \times 224$ Foot pipeline resolution.
To prevent input resolution from acting as an uncontrolled experimental variable (e.g. evaluating EfficientNet-B3 at $300 \times 300$ vs ViT at $224 \times 224$), all architectures receive identical $224 \times 224 \times 3$ RGB inputs.

---

## Model Inspection & Profiling

| Model Name | Backbone Class | Total Params (M) | Model Size (MB) | Head Dropout | Weight Source | Fixed Input Resolution |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: |
| **EfficientNet-B0** | `FootEfficientNetB0` | ~4.01M | ~15.3 MB | 0.2 | ImageNet DEFAULT | $224 \times 224$ |
| **EfficientNet-B3** | `FootEfficientNetB3` | ~10.70M | ~40.8 MB | 0.3 | ImageNet DEFAULT | $224 \times 224$ |
| **ConvNeXt-Tiny** | `FootConvNeXtTiny` | ~27.82M | ~106.1 MB | 0.2 | ImageNet DEFAULT | $224 \times 224$ |
| **Swin-Tiny** | `FootSwinTiny` | ~27.52M | ~105.0 MB | 0.2 | ImageNet DEFAULT | $224 \times 224$ |
| **ViT-B/16** | `FootViTB16` | ~85.80M | ~327.3 MB | 0.2 | ImageNet DEFAULT | $224 \times 224$ |

All models are instantiated via central model factory `src/foot/models/factory.py` using `MODEL_REGISTRY`.
