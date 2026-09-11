# 08 — Source Group Analysis & Leakage Investigation

## 1. Offline Augmentation Expansion Discovery
Script `src/foot/data/investigate_leakage.py` extracted root source image IDs (`source_image_id`) by splitting Roboflow augmentation hashes (`.rf.<hash>`).

| Group Metric | Measured Value |
| :--- | :--- |
| **Total Raw Image Files** | **10,062** |
| **Unique Source Image Groups** | **1,770** |
| **Offline Expansion Ratio** | **$5.68\times$** (1,770 source images expanded into 10,062 variants) |
| **Source Image Cross-Split Leakage** | 10 source image groups span across raw `train` and `valid`/`test` folders |

## 2. Patient Identifier Availability
- **Patient IDs**: `UNAVAILABLE` in raw metadata.
- **Scientific Limitation**: Patient-level separation cannot be guaranteed because patient IDs are absent. Group-stratified partitioning on `source_image_id` is enforced as the primary control mechanism.
