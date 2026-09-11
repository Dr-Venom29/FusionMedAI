# 02 — Exact Duplicate Resolution (Phase 10.2.1)

## 1. Methodology & Selection Rule
Script `src/foot/data/resolve_exact_duplicates.py` processed all 10,062 raw image files to identify exact SHA-256 byte duplicates.

- **Selection Rule**: Deterministic partition priority (`train` > `valid` > `test`), followed by lexicographically lowest relative file path.

## 2. Quantitative Results

| Resolution Category | Count | Status / Role |
| :--- | :---: | :--- |
| **Total Scanned Images** | **10,062** | Full raw inventory |
| **Unique SHA-256 Hashes** | **10,050** | Re-computed & verified cryptographic hashes |
| **Canonical Images Kept** | **10,050** | `status == "KEPT"`, `role == "CANONICAL"` |
| **Exact Duplicates Excluded** | **12** | `status == "EXCLUDED"`, `role == "EXCLUDED_EXACT_DUPLICATE"` |
| **Exact Duplicate Groups** | **12** | 12 pairs of byte-exact duplicate files |
| **Cross-Class Exact Conflicts** | **0** | **0% label conflicts** across exact duplicate groups |

## 3. Output Artifacts
- **Script**: [src/foot/data/resolve_exact_duplicates.py](file:///d:/FusionMedAI/src/foot/data/resolve_exact_duplicates.py)
- **Report**: [datasets/foot/metadata/duplicate_resolution.json](file:///d:/FusionMedAI/datasets/foot/metadata/duplicate_resolution.json)
- **Manifest**: [datasets/foot/metadata/statistics/foot_duplicate_resolution.csv](file:///d:/FusionMedAI/datasets/foot/metadata/statistics/foot_duplicate_resolution.csv)
