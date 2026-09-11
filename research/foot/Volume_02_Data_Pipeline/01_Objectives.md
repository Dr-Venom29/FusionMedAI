# 01 — Data Pipeline Objectives & Requirements

## 1. Objectives
The objective of Phase 10.2 is to transform the frozen, audited Foot DFU dataset into a leakage-controlled, reproducible PyTorch data pipeline without altering raw dataset files in `datasets/foot/raw/`.

## 2. Requirements
1. **Raw Immutability**: Keep `datasets/foot/raw/` 100% read-only.
2. **Exact Duplicate Resolution**: Exclude 12 exact byte-level duplicates from the modeling population.
3. **Source Grouping**: Group all augmented variants sharing the same `source_image_id` into indivisible units.
4. **0% Cross-Split Group Leakage**: Enforce group-stratified partitioning across train (80%), validation (10%), and test (10%) splits.
5. **Reproducible Pipeline**: Implement PyTorch Dataset, transforms, and DataLoaders supporting deterministic validation/testing and reproducible seeding (`SEED = 42`).
