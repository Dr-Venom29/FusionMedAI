# 03 Class Distribution — Phase 10.3.2

## 1. Canonical Population Distribution

The canonical population consists of **10,050 images** grouped into **1,770 source groups**.

| Wagner Grade | Description | Code | Canonical Images | Image % | Source Groups | Group % |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Grade 1** | Superficial Ulcer | 0 | **2,367** | 23.55% | **449** | 25.37% |
| **Grade 2** | Deep Ulcer to Tendon/Bone | 1 | **2,457** | 24.45% | **443** | 25.03% |
| **Grade 3** | Deep Ulcer with Osteomyelitis | 2 | **2,797** | 27.83% | **437** | 24.69% |
| **Grade 4** | Partial Foot Gangrene | 3 | **2,429** | 24.17% | **441** | 24.92% |
| **Total** | | | **10,050** | **100.00%** | **1,770** | **100.00%** |

### Imbalance Ratio
- **Max Class**: Grade 3 (2,797 images, 27.83%)
- **Min Class**: Grade 1 (2,367 images, 23.55%)
- **Imbalance Ratio**: $\frac{2797}{2367} = \mathbf{1.18:1}$ (Near-perfect balance)

---

## 2. Partition Preservation

Group-stratified partitioning preserved class proportions across all three partitions:

| Wagner Grade | Code | Train % (8,038) | Val % (1,006) | Test % (1,006) | Overall % (10,050) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Grade 1** | 0 | 23.55% | 23.56% | 23.56% | 23.55% |
| **Grade 2** | 1 | 24.45% | 24.45% | 24.45% | 24.45% |
| **Grade 3** | 2 | 27.83% | 27.83% | 27.83% | 27.83% |
| **Grade 4** | 3 | 24.17% | 24.16% | 24.16% | 24.17% |
