# 01 — Audit Objectives & Scope

## 1. Audit Rationale
Prior to training machine learning models for clinical Diabetic Foot Ulcer (DFU) grading, the raw dataset must undergo rigorous empirical validation. Training models on unverified medical image collections introduces risks of data leakage, label misalignment, undetected corruption, and unstated dataset expansion artifacts.

## 2. Core Objectives
1. **Inventory & Integrity Verification**: Confirm that all distributed files are valid, decodable JPEGs with correct channel dimensions.
2. **Label Verification**: Establish explicit, verified folder-to-class mappings based on the Wagner–Meggitt grading scale.
3. **Property Characterization**: Measure observed color channel intensity distributions (mean and standard deviation) without assuming natural image defaults.
4. **Duplicate & Leakage Investigation**: Identify exact byte-level duplicates (SHA-256) and perceptual near-duplicates (dHash) across splits and classes.
5. **Source-Image Grouping**: Analyze offline dataset expansion to reconstruct underlying source image groups.
6. **Formal Quality Decision**: Execute a fail-fast dataset quality assessment (PASS, CONDITIONAL PASS, or FAIL) prior to pipeline engineering.

## 3. Guiding Principles
- **Immutable Raw Data**: Raw dataset files under `datasets/foot/raw/` are treated as strictly read-only.
- **Empirical Measurement**: Audit conclusions must derive directly from executed python audit scripts and persistent metadata records.
- **Transparent Limitation Recording**: Document dataset constraints (such as missing patient IDs) rather than assuming ideal conditions.
