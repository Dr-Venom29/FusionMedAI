# Phase 10.3.7 — Final Foot EDA & Quality Decision Report

## 1. Executive Summary & Acceptance Gate

┌──────────────────────────────────────────────────────────┐
│             FOOT EDA & QUALITY ACCEPTANCE GATE           │
├──────────────────────────────────────────────────────────┤
│ Dataset Characteristics Understood               PASS    │
│ Quality Issues Classified                        PASS    │
│ Potential Biases Documented                      PASS    │
│ Zero Unjustified Image Deletions                 PASS    │
│ Preprocessing & Loss Policy Finalized            PASS    │
│ FINAL EDA ACCEPTANCE GATE                        PASS    │
└──────────────────────────────────────────────────────────┘

Phase 10.3 (Foot Exploratory Data Analysis & Dataset Quality) is **100% complete and fully verified**. The modeling population consists of **10,050 canonical images** across **1,770 source groups** partitioned into Train (8,038), Val (1,006), and Test (1,006).

---

## 2. Defensible Research Trail ("We observed X, therefore we decided Y because Z")

### 1. Normalization Policy
- **Observation (X)**: Measured exact pixel RGB statistics across 10,050 canonical images: Mean=$[0.4937, 0.3630, 0.3272]$, Std=$[0.1745, 0.1632, 0.1551]$.
- **Decision (Y)**: Use observed dataset normalization statistics rather than standard ImageNet defaults.
- **Rationale (Z)**: Red channel mean ($0.4937$) is significantly higher than Blue ($0.3272$) due to tissue hyperemic redness and vascularity. Custom normalization centers color distributions accurately for clinical feature extraction.

### 2. Outlier Retention Policy
- **Observation (X)**: Identified 1,414 quality outliers (14.07%) including soft background blur (6.39%), low contrast (4.37%), and color cast (3.13%).
- **Decision (Y)**: Retain 100% of outlier images in the modeling dataset without deletion.
- **Rationale (Z)**: Deleting outliers would distort patient source-group integrity and introduce artificial data selection bias. Models must be evaluated under real-world clinical photography conditions.

### 3. Backbone & Loss Function Selection
- **Observation (X)**: Pre-trained ResNet50 feature embeddings yielded a near-zero Silhouette score ($-0.0004$) and minimum centroid distance ($0.0193$) between Grade 2 and Grade 3.
- **Decision (Y)**: Mandate supervised end-to-end fine-tuning of deep backbones (ConvNeXt, Swin, EfficientNet) and prioritize Macro F1 & Grade 3 Recall evaluation metrics.
- **Rationale (Z)**: Grade 2 (deep ulcer without osteomyelitis) and Grade 3 (deep ulcer with bone sepsis) share high surface visual similarity. Generic pre-trained features cannot separate them without supervised fine-tuning.

### 4. Class Imbalance & Loss Weighting Policy
- **Observation (X)**: Observed near-perfect class balance (Imbalance Ratio $1.18:1$; Grade 1: 23.55%, Grade 2: 24.45%, Grade 3: 27.83%, Grade 4: 24.17%).
- **Decision (Y)**: Use standard epoch-based random shuffling with Cross-Entropy Loss or Sqrt Inverse Frequency weights (Grade 1: 1.0870, Grade 2: 1.0669, Grade 3: 1.0000, Grade 4: 1.0731).
- **Rationale (Z)**: Heavy oversampling is unnecessary and risks overfitting to specific patient source groups.

### 5. Data Augmentation Policy
- **Observation (X)**: Audited non-clinical features against Wagner grade; brightness ($r=-0.1441$) and Roboflow naming prefixes ($V=0.1316$) show low shortcut risk, while RMS contrast ($r=+0.2406$) shows moderate correlation due to necrotic eschar contrast.
- **Decision (Y)**: Apply random color jitter (brightness=$0.2$, contrast=$0.2$, saturation=$0.1$) and affine rotation ($\pm 15^\circ$) during training.
- **Rationale (Z)**: Color jitter destroys potential residual illumination/color shortcuts without obscuring true tissue necrosis boundaries.

---

## 3. Finalized Preprocessing & Pipeline Directives

- **Input Resolution**: $224 	imes 224 	imes 3$ RGB (100.00% uniform)
- **Normalization**: Mean=$[0.4937, 0.3630, 0.3272]$, Std=$[0.1745, 0.1632, 0.1551]$
- **Train Augmentation**: Horizontal Flip ($p=0.5$), Rotation ($\pm 15^\circ$), Color Jitter ($b=0.2, c=0.2, s=0.1$)
- **Validation / Test Policy**: Strictly deterministic (Resize + Normalization only; 0% augmentation leakage)
- **Loss Function**: Weighted Cross-Entropy or Focal Loss ($\gamma = 2.0$)
- **Primary Metric**: Macro F1-Score

---

## 4. Formal Sign-Off

Phase 10.3 is **PASSED**. The repository is authorized to proceed to **Phase 10.4 — Foot Model Training & Baseline Benchmarking**.
