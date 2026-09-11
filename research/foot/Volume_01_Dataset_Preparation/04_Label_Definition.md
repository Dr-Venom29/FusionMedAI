# 04 — Wagner Grade Label Definitions

## 1. Wagner–Meggitt Classification System

The dataset categorizes Diabetic Foot Ulcers into four grades based on the Wagner–Meggitt clinical classification system.

| Raw Folder Name | Class Index | Clinical Wagner Grade & Description |
| :--- | :---: | :--- |
| `Grade 1` | `0` | **Superficial Ulcer**: Full skin thickness loss without subcutaneous tissue involvement. |
| `Grade 2` | `1` | **Deep Ulcer**: Deep tissue penetration extending to tendon, ligament, or joint capsule, *without* bone involvement or abscess. |
| `Grade 3` | `2` | **Deep Ulcer with Complications**: Deep ulcer with tissue abscess, osteomyelitis (bone infection), or joint sepsis. |
| `Grade 4` | `3` | **Localized Gangrene**: Gangrenous necrosis localized to forefoot or heel. |

## 2. Verification Outcomes
- Script `src/foot/data/verify_labels.py` confirmed 100% folder-to-class alignment.
- Zero unexpected directory names, hidden system files (`.DS_Store`, `Thumbs.db`), or duplicate class representations were present in `datasets/foot/raw/`.
