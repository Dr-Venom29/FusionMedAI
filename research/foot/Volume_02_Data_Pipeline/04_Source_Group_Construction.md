# 04 — Source-Image Group Construction (Phase 10.2.3)

## 1. Grouping Rationale & Extraction
Script `src/foot/data/construct_source_groups.py` extracted root `source_image_id` values prior to Roboflow augmentation hash patterns (`.rf.<hash>`).

## 2. Quantitative Grouping Summary

| Metric | Measured Value |
| :--- | :---: |
| **Total Source-Image Groups** | **1,770** |
| **Total Canonical Images Grouped** | **10,050** |
| **Total Raw Images Grouped** | **10,062** |
| **Mean Variants per Group** | **5.68** |
| **Variant Count Range** | **1 to 21** |

## 3. Group-Level Class Breakdown

| Wagner Grade | Class Name | Group Count | Group Percentage |
| :--- | :--- | :---: | :---: |
| **Grade 1** | Superficial Ulcer | **449** | 25.37% |
| **Grade 2** | Deep Ulcer | **443** | 25.03% |
| **Grade 3** | Abscess / Osteomyelitis | **437** | 24.69% |
| **Grade 4** | Localized Gangrene | **441** | 24.91% |
| **Total** | | **1,770** | **100.00%** |
