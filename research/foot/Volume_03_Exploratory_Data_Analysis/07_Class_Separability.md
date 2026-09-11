# 07 Class Separability Analysis — Phase 10.3.6

## 1. Feature Space & Dimensionality Reduction

Class separability was evaluated by extracting 2,048-dimensional ImageNet pre-trained ResNet50 feature embeddings across all **10,050 canonical images**.

- **PCA Variance**: Top 2 Principal Components explain **18.04%** variance; Top 50 PCs explain **67.45%** variance.
- **Silhouette Score** (PCA 50-dim): **`-0.0004`** (Confirms continuous feature spectrum without artificial isolated cluster gaps).
- **5-NN Class Agreement**: **`97.08%`**
- **10-NN Class Agreement**: **`94.98%`**

---

## 2. Pairwise Centroid Cosine Distances

| Wagner Class Comparison Pair | Cosine Distance | Separation Level | Clinical Correlation |
| :--- | :---: | :--- | :--- |
| **Grade 1 vs Grade 4** | **`0.0631`** | **MAXIMUM SEPARATION** | Superficial red lesion vs Charcoal gangrene eschar. |
| **Grade 2 vs Grade 4** | **`0.0420`** | High Separation | Deep non-gangrenous ulcer vs Ischemic gangrene. |
| **Grade 1 vs Grade 3** | **`0.0349`** | High Separation | Superficial erosion vs Deep ulcer with bone sepsis. |
| **Grade 1 vs Grade 2** | **`0.0221`** | Moderate Separation | Superficial skin erosion vs Subcutaneous penetration. |
| **Grade 3 vs Grade 4** | **`0.0204`** | Moderate Separation | Purulent osteomyelitis vs Localized gangrene eschar. |
| **Grade 2 vs Grade 3** | **`0.0193`** | **MINIMUM DISTANCE / MAXIMUM OVERLAP** | Deep ulcer without osteomyelitis vs Deep ulcer with osteomyelitis. |

---

## 3. Modeling Implications

1. Supervised end-to-end fine-tuning is required to learn fine-grained tissue textures (slough, sinus tracts) beyond generic ImageNet representations.
2. PCA & t-SNE scatter plots are exported to `datasets/foot/metadata/eda/figures/class_separability_pca_tsne.png`.
