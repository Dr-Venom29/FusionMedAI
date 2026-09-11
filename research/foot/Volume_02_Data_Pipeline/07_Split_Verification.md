# 07 — Split Verification & Leakage Audit (Phase 10.2.6)

## 1. Leakage & Integrity Audit Results
Script `src/foot/data/verify_split.py` verified the split CSV manifests.

| Audit Check | Count Detected | Status |
| :--- | :---: | :---: |
| **Source Groups Shared Across Splits** | `0` | ✅ PASS |
| **Exact Hashes Shared Across Splits** | `0` | ✅ PASS |
| **TYPE 1 Near-Duplicates Shared Across Splits** | `0` | ✅ PASS |
| **Missing Image Files on Disk** | `0` | ✅ PASS |
| **Invalid Wagner Grade Labels** | `0` | ✅ PASS |
| **Duplicate Manifest Rows** | `0` | ✅ PASS |
| **Corrupted / Unreadable Images** | `0` | ✅ PASS |

## 2. Conclusion
0% group leakage verified across all split boundaries. Data manifests are 100% valid and decodable.
