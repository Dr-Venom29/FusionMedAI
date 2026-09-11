# 05 — Data Integrity & Decodability Audit

## 1. Integrity Verification Methodology
The image integrity audit script (`src/foot/data/audit_integrity.py`) executed image decodability, header parsing, byte truncation checks, and channel structure verification across all 10,062 image files using PIL Image decoders.

## 2. Integrity Audit Results

| Audit Status Category | Image Count | Percentage | Verification Result |
| :--- | :---: | :---: | :--- |
| **VALID** | **10,062** | **100.00%** | Decodes cleanly, $224 \times 224$ RGB, non-truncated |
| **CORRUPT** | `0` | `0.00%` | Zero undecodable files detected |
| **UNREADABLE** | `0` | `0.00%` | Zero 0-byte files detected |
| **INVALID FORMAT** | `2` | N/A | Non-image text files (`README.dataset.txt`, `README.roboflow.txt`) |

## 3. Conclusion
All 10,062 image files in `datasets/foot/raw/` are valid, un-truncated, 3-channel RGB JPEGs.
