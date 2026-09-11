# 02 — Dataset Selection & Ingestion Layout

## 1. Primary Source & Access
The Diabetic Foot Ulcer (DFU) dataset utilized in this modality is the **ADPM V3.3 Classification** dataset, accessed via Kaggle mirror (`DFU_Dataset_annotated_into_4_classes`) and original Roboflow Universe export (`https://universe.roboflow.com/adpm/adpm-v3.3-classification`).

## 2. Directory Layout
The raw dataset is stored in `datasets/foot/raw/` with the following structure:

```directory
datasets/foot/raw/
├── README.dataset.txt
├── README.roboflow.txt
├── train/
│   ├── Grade 1/
│   ├── Grade 2/
│   ├── Grade 3/
│   └── Grade 4/
├── valid/
│   ├── Grade 1/
│   ├── Grade 2/
│   ├── Grade 3/
│   └── Grade 4/
└── test/
    ├── Grade 1/
    ├── Grade 2/
    ├── Grade 3/
    └── Grade 4/
```

## 3. Storage Principles
- **Read-Only Ingestion**: Files in `datasets/foot/raw/` remain unmodified. All downstream preprocessing, manifest generation, and index splitting occur programmatically under `datasets/foot/processed/` and `src/foot/data/`.
