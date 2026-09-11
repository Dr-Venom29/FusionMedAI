# 08 — Foot Dataset Class Implementation (Phase 10.2.7)

## 1. Class Architecture
Module [src/foot/data/dataset.py](file:///d:/FusionMedAI/src/foot/data/dataset.py) defines the `FootDFUDataset` PyTorch `Dataset` class.

## 2. Key Features & Interface
- **Lazy Loading**: Opens images on demand with `PIL.Image.open().convert("RGB")`.
- **Flexible Return Types**:
  - Default (`return_metadata=False`): Returns `(image, label)` tuple.
  - Metadata mode (`return_metadata=True`): Returns `(image, label, metadata)` tuple.
- **Verification**: Unit tests confirmed sample loading for `train.csv`, `val.csv`, and `test.csv`.
