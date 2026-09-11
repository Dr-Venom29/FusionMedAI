# 07 — Duplicate & Near-Duplicate Audit

## 1. Exact Duplicate Audit (SHA-256)
Script `src/foot/data/detect_duplicates.py` computed cryptographic SHA-256 hashes across all 10,062 image files.

- **Unique SHA-256 Hashes**: 10,050
- **Exact Duplicate Groups**: 12 groups (24 total files)
- **Exact Cross-Split Leakage**: 0 byte-exact duplicates shared across raw train/val/test splits.
- **Exact Cross-Class Label Conflicts**: 0 byte-exact duplicates assigned to different classes.

## 2. Near-Duplicate Audit (dHash Distance $\le 4$)
Perceptual hash analysis using difference hashing (dHash 64-bit) identified:
- **Total Near-Duplicate Pairs (dHash $\le 4$)**: 1,964 pairs (in raw audit) / 2,791 pairs (in canonical set)
- **Near Cross-Split Pairs**: 5 pairs found in raw subfolder splits (`train` vs `valid`/`test`).
- **Near Cross-Class Pairs**: 114 pairs assigned across different Wagner grades.

## 3. Impact & Policy
Exact duplicate files (12 copies) must be resolved by keeping 1 canonical image per hash. Near-duplicates represent offline augmented variants of the same underlying photography requiring group-level isolation.
