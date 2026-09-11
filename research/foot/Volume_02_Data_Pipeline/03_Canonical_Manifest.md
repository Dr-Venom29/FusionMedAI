# 03 — Canonical Dataset Manifest (Phase 10.2.2)

## 1. Primary Manifest Overview
Script `src/foot/data/create_canonical_manifest.py` constructed the single authoritative modeling manifest [canonical_manifest.csv](file:///d:/FusionMedAI/datasets/foot/processed/canonical_manifest.csv) under `datasets/foot/processed/`.

## 2. Manifest Fields & Schema

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `id_code` | `string` | Unique record identifier (filename stem) |
| `image_path` | `string` | Relative path from `datasets/foot/raw/` |
| `source_image_id` | `string` | Extracted root source image identifier |
| `wagner_grade` | `integer` | Numerical Wagner grade index (`0..3`) |
| `class_name` | `string` | Human-readable Wagner grade description (`Grade 1..4`) |
| `original_partition` | `string` | Raw subfolder partition (`train`, `valid`, `test`) |
| `sha256` | `string` | Cryptographic SHA-256 hash |
| `is_canonical` | `boolean` | Modeling flag (`True` = Kept canonical, `False` = Excluded duplicate) |

## 3. Manifest Statistics
- **Total Rows**: 10,062
- **Canonical Rows (`is_canonical == True`)**: 10,050
- **Excluded Rows (`is_canonical == False`)**: 12
- **Unique Source Groups (`source_image_id`)**: 1,770
