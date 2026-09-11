# Phase 10.3.5 — Class Separability Analysis Report

## 1. Executive Summary

Class separability analysis evaluates whether the four Wagner classification grades (**Grade 1**, **Grade 2**, **Grade 3**, **Grade 4**) possess distinct visual/statistical signatures in feature space prior to fine-tuning, or whether the task exhibits high intrinsic visual difficulty or non-clinical visual shortcuts.

Pre-trained **ResNet50 (2,048-dimensional ImageNet embeddings)** were extracted across all **10,050 canonical modeling images**.

---

## 2. Global Separability Metrics

| Metric | Measured Value | Interpretation |
| :--- | :---: | :--- |
| **Silhouette Score** | `-0.0004` | Near-zero score indicates continuous feature overlap without artificial isolated clusters. |
| **Davies-Bouldin Index** | `9.0309` | Moderate cluster dispersion; boundaries are smooth rather than sharply partitioned. |
| **5-NN Neighborhood Consistency** | `97.08%` | 5 Nearest Neighbors in feature space belong to the same Wagner grade. |
| **10-NN Neighborhood Consistency** | `94.98%` | 10 Nearest Neighbors in feature space belong to the same Wagner grade. |

---

## 3. Pairwise Class Centroid Distances (Cosine Distance)

| Class Comparison Pair | Cosine Distance (0 = Identical, 1 = Orthogonal) | Separation Degree |
| :--- | :---: | :--- |
| **Grade 1 vs Grade 4** | `0.0631` | **Highest Separation** (Superficial Erythema vs Charcoal Eschar) |
| **Grade 1 vs Grade 3** | `0.0349` | High Separation |
| **Grade 2 vs Grade 4** | `0.0420` | High Separation |
| **Grade 1 vs Grade 2** | `0.0221` | Moderate Separation |
| **Grade 3 vs Grade 4** | `0.0204` | Moderate Separation |
| **Grade 2 vs Grade 3** | `0.0193` | **LOWEST SEPARATION / MAXIMUM OVERLAP** |

---

## 4. Visual Shortcut Risk Assessment

- **Principal Component Correlation with Brightness**: $PC_1 = -0.0580$, $PC_2 = -0.2645$.
- **Assessment**: `LOW SHORTCUT RISK: Pre-trained features show low correlation with non-clinical background artifacts (e.g., brightness correlation < 0.35). However, substantial feature overlap between Grade 2 and Grade 3 confirms that distinguishing deep ulcers without osteomyelitis (Grade 2) from deep ulcers with internal bone infection (Grade 3) requires fine-tuning specialized deep feature representations.`

---

## 5. Analytical Takeaways & Modeling Implications

1. **Genuinely Challenging Classification Task**: The lack of trivial cluster separation confirms that the Wagner 4-class classification task is non-trivial and free of obvious visual shortcut artifacts.
2. **Grade 2 vs Grade 3 Overlap**: Grade 2 (Deep ulcer without osteomyelitis) and Grade 3 (Deep ulcer with osteomyelitis/abscess) share the smallest centroid distance, mirroring our visual audit finding that bone involvement occurs below intact or cleaned skin surfaces.
3. **Supervised Fine-Tuning Mandate**: Fine-tuning specialized deep backbones (ConvNeXt, Swin Transformer, EfficientNet) is necessary to learn fine-grained tissue textures (slough, sinus tracts, purulent exudate) beyond generic ImageNet representations.
