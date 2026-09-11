# 09 Findings & Defensible Research Trail — Phase 10.3.8

## 1. Defensible Research Trail ("We observed X, therefore we decided Y because Z")

### 1. Custom Normalization Policy
- **We observed (X)**: Measured exact pixel RGB statistics across 10,050 canonical images: Mean=$[0.4937, 0.3630, 0.3272]$, Std=$[0.1745, 0.1632, 0.1551]$.
- **Therefore we decided (Y)**: Use observed dataset normalization statistics rather than standard ImageNet defaults.
- **Because (Z)**: Red channel mean ($0.4937$) is significantly higher than Blue ($0.3272$) due to tissue hyperemic redness and vascularity. Custom normalization centers color distributions accurately for clinical feature extraction.

### 2. Zero Outlier Deletion Policy
- **We observed (X)**: Identified 1,414 quality outliers (14.07%) including soft background blur (6.39%), low contrast (4.37%), and color cast (3.13%).
- **Therefore we decided (Y)**: Retain 100% of outlier images in the modeling dataset without deletion.
- **Because (Z)**: Deleting outliers would distort patient source-group integrity and introduce artificial data selection bias. Models must be evaluated under real-world clinical photography conditions.

### 3. Supervised Deep Fine-Tuning Mandate
- **We observed (X)**: Pre-trained ResNet50 feature embeddings yielded a near-zero Silhouette score ($-0.0004$) and minimum centroid distance ($0.0193$) between Grade 2 and Grade 3.
- **Therefore we decided (Y)**: Mandate supervised end-to-end fine-tuning of deep backbones (ConvNeXt, Swin, EfficientNet) and prioritize Macro F1 & Grade 3 Recall evaluation metrics.
- **Because (Z)**: Grade 2 (deep ulcer without osteomyelitis) and Grade 3 (deep ulcer with bone sepsis) share high surface visual similarity. Generic pre-trained features cannot separate them without supervised fine-tuning.

### 4. Class Balance & Loss Weighting Policy
- **We observed (X)**: Observed near-perfect class balance (Imbalance Ratio $1.18:1$; Grade 1: 23.55%, Grade 2: 24.45%, Grade 3: 27.83%, Grade 4: 24.17%).
- **Therefore we decided (Y)**: Use standard epoch-based random shuffling with Cross-Entropy Loss or Sqrt Inverse Frequency weights (Grade 1: 1.0870, Grade 2: 1.0669, Grade 3: 1.0000, Grade 4: 1.0731).
- **Because (Z)**: Heavy oversampling is unnecessary and risks overfitting to specific patient source groups.

### 5. Color Jitter Augmentation Policy
- **We observed (X)**: Audited non-clinical features against Wagner grade; brightness ($r=-0.1441$) and Roboflow naming prefixes ($V=0.1316$) show low shortcut risk, while RMS contrast ($r=+0.2406$) shows moderate correlation due to necrotic eschar contrast.
- **Therefore we decided (Y)**: Apply random color jitter (brightness=$0.2$, contrast=$0.2$, saturation=$0.1$) and affine rotation ($\pm 15^\circ$) during training.
- **Because (Z)**: Color jitter destroys potential residual illumination/color shortcuts without obscuring true tissue necrosis boundaries.
