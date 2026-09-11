# Phase 10.2.6 — Split Verification Report

```text
================================================================================
SPLIT VERIFICATION STATUS: PASS
All leakage, integrity, and class distribution checks passed cleanly.
================================================================================
```

## 1. Leakage Verification Results

| Leakage Category | Count Detected | Status |
| :--- | :---: | :---: |
| **Source Groups Shared Across Splits** | `0` | ✅ PASS |
| **Exact Duplicates Across Splits** | `0` | ✅ PASS |
| **Known TYPE 1 Near-Duplicates Across Splits** | `0` | ✅ PASS |

---

## 2. Data Integrity Results

| Integrity Check | Count Detected | Status |
| :--- | :---: | :---: |
| **Missing Image Files on Disk** | `0` | ✅ PASS |
| **Invalid Wagner Grade Labels** | `0` | ✅ PASS |
| **Duplicate Manifest Rows** | `0` | ✅ PASS |
| **Corrupted / Unreadable Images** | `0` | ✅ PASS |

---

## 3. Stratified Class & Group Distribution

| Partition Split | Total Images | Total Groups | Grade 1 | Grade 2 | Grade 3 | Grade 4 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | **8,038** | **1,412** | 1,893 (23.55%) | 1,965 (24.45%) | 2,237 (27.83%) | 1,943 (24.17%) |
| **Validation** | **1,006** | **177** | 237 (23.56%) | 246 (24.45%) | 280 (27.83%) | 243 (24.16%) |
| **Test** | **1,006** | **181** | 237 (23.56%) | 246 (24.45%) | 280 (27.83%) | 243 (24.16%) |
| **Total** | **10,050** | **1,770** | **2,367** | **2,457** | **2,797** | **2,429** |

---

## Conclusion
The group-stratified train/val/test splits are verified as **leakage-free**, **fully decodable**, and **properly stratified**. The dataset split is approved for DataLoader implementation (`Phase 10.2.7`).
