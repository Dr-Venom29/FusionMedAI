# FusionMedAI Datasets

This directory contains the datasets used in FusionMedAI for multimodal
diabetic disease analysis across three planned modalities:

- Retina fundus images
- Diabetic foot-ulcer images
- Clinical tabular data

Each modality maintains an independent dataset directory and research
pipeline.

---

## Folder Structure

```text
datasets/

├── retina/
│   ├── raw/                  # Raw APTOS 2019 data (immutable)
│   ├── interim/
│   ├── processed/
│   └── metadata/

├── foot/
│   ├── raw/                  # Raw DFU Wagner data (immutable)
│   ├── interim/
│   ├── processed/
│   └── metadata/

└── README.md