# 09 — Candidate Image Transforms (Phase 10.2.8)

## 1. Module Architecture
Module [src/foot/data/transforms.py](file:///d:/FusionMedAI/src/foot/data/transforms.py) provides torchvision transformation functions.

## 2. Evaluation & Clinical Rationale

| Transformation | Parameters | Clinical & Data Rationale | Policy |
| :--- | :---: | :--- | :---: |
| **Normalization** | Mean `[0.4937, 0.3630, 0.3272]`<br>Std `[0.1744, 0.1632, 0.1551]` | Uses observed Foot DFU dataset statistics. | **APPROVED** |
| **Resizing** | `$224 \times 224$` | Standardizes input tensor resolution. | **APPROVED** |
| **Random Rotation** | $\pm 15^\circ$ | Simulates camera roll and leg positioning variance in clinical photography. | **APPROVED** |
| **Random Horizontal Flip** | $p = 0.5$ | Evaluated for laterality; Wagner grade relies on ulcer tissue depth, which is invariant under horizontal reflection. | **APPROVED** |
| **Color Jitter** | Brightness (0.2), Contrast (0.2), Saturation (0.1), Hue (0.02) | Simulates varying ambient clinic lighting and mobile sensor white balances. | **APPROVED** |
