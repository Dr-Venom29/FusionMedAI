# Phase 10.1.F — Duplicate Detection & Leakage Audit Report

## Executive Summary
- **Total Images Audited**: `10,062`
- **Unique Exact Images**: `10,050`
- **Exact Duplicate Groups**: `12`
- **Exact Duplicate Files Count**: `24`
- **Near Duplicate Pairs (dHash $\le 4$)**: `1964`

## Data Leakage & Label Conflict Audit
| Audit Check | Count Detected | Status | Impact / Risk Assessment |
| :--- | :---: | :---: | :--- |
| **Exact Cross-Split Leakage** | `0` | ✅ PASS | Duplicate image exists in both train and test/valid splits |
| **Exact Cross-Class Conflict** | `0` | ✅ PASS | Identical image assigned to multiple different Wagner grades |
| **Near Cross-Split Leakage** | `5` | ⚠️ LEAK DETECTED | Visually identical image across train/test partitions |
| **Near Cross-Class Conflict** | `114` | ⚠️ CONFLICT DETECTED | Visually identical image assigned to different Wagner grades |

## Rule Compliance & Immutability Notice
> **Immutability Enforced**: No files were removed or altered inside `datasets/foot/raw/`. Duplicate groups and leakage instances are recorded in `duplicate_report.json` and `foot_duplicate_pairs.csv` to inform downstream dataset split construction in Step 10.2.