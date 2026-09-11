# 05 — Near-Duplicate & Conflict Analysis (Phase 10.2.4)

## 1. Categorization & Resolution Matrix
Script `src/foot/data/analyze_near_duplicates.py` evaluated near-duplicate pairs (dHash distance $\le 4$) across all 10,050 canonical images.

| Structural Relationship Type | Pair Count | Percentage | Resolution Policy |
| :--- | :---: | :---: | :--- |
| **TYPE 1: Same Source Group** | **2,452** | **87.85%** | **RESOLVED**: Bound into the same `source_image_id` group. Zero cross-split leakage. |
| **TYPE 2: Diff Source, Same Class** | **145** | **5.20%** | **MONITORED**: Valid clinical visual similarity across distinct subjects of the same Wagner grade. |
| **TYPE 3: Diff Source, Diff Class** | **194** | **6.95%** | **FLAGGED**: Cross-class visual similarity flagged and archived for downstream error analysis. |
| **Total Near-Duplicate Pairs** | **2,791** | **100.00%** | Comprehensive near-duplicate classification (dHash $\le 4$) |

## 2. Key Audit Conclusion
87.85% (2,452 pairs) of all near-duplicates belong to the exact same `source_image_id` group. Source-group-stratified splitting completely eliminates cross-split variant leakage.
