# Phase 10.3.2 — Class-Wise Visual Analysis Report

## 1. Overview & Class Profiles

| Wagner Class | Sample Count | Mean Luminance | Red Mean | Green Mean | Blue Mean | Mean File Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Grade 1** (Superficial Ulcer) | 2,367 | `0.4299` | `0.5408` | `0.3878` | `0.356` | `6.3 KB` |
| **Grade 2** (Deep Ulcer) | 2,457 | `0.3929` | `0.5032` | `0.35` | `0.3246` | `6.6 KB` |
| **Grade 3** (Abscess / Osteo) | 2,797 | `0.3881` | `0.4744` | `0.3576` | `0.3188` | `6.79 KB` |
| **Grade 4** (Gangrene) | 2,429 | `0.3832` | `0.4602` | `0.358` | `0.3112` | `7.46 KB` |

---

## 2. Visual Characteristics & Class Overlap Assessment

### Grade 1 vs Grade 2
- **Differentiating Feature**: Grade 1 presents superficial epidermal/dermal skin erosion. Grade 2 penetrates deep to tendon, ligament, or joint capsule.
- **Visual Overlap**: Moderate overlap when ulcer bed depth is obscured by slough or camera angle.

### Grade 2 vs Grade 3 (CRITICAL CLINICAL OVERLAP)
- **Differentiating Feature**: Grade 2 is a deep ulcer *without* bone involvement or abscess. Grade 3 involves osteomyelitis (bone infection) or abscess.
- **Visual Overlap**: **HIGH VISUAL OVERLAP**. Because osteomyelitis resides internally within bone structures, Grade 2 and Grade 3 ulcers often present near-identical surface photography unless purulent exudate or sinus drainage is visually prominent.

### Grade 3 vs Grade 4
- **Differentiating Feature**: Grade 3 features purulent infection/abscess. Grade 4 features dry or wet black ischemic gangrene.
- **Visual Overlap**: Low to moderate. Distinct black necrotic eschar in Grade 4 provides a strong visual signature.

---

## 3. Visual Grid Artifact
- **Comparison Grid**: Archived at `datasets/foot/metadata/quality/wagner_class_comparison_grid.png`.
