# 02 Benchmark Protocol — Phase 10.5

## 10.5.3 Dataset and Frozen Experimental Population

- **Dataset**: ADPM V3.3 / 4-Class Wagner Diabetic Foot Ulcer Classification
- **Canonical Images**: 10,050
- **Source-Image Groups**: 1,770
- **Original Valid Images**: 10,062
- **Deduplication**: After exact-duplicate resolution, 10,050 canonical images were retained.

*Note: Patient identifiers are unavailable in the public dataset; therefore, patient-level separation cannot be independently verified. Source-image grouping is used to prevent known source-level data leakage.*

---

## 10.5.4 Frozen Data Split

The benchmark uses the frozen Phase 10.2 source-group-stratified split.

| Split | Source Groups | Images | Percentage |
|---|---:|---:|---:|
| **Train** | 1,412 | 8,038 | 80.0% |
| **Validation** | 177 | 1,006 | 10.0% |
| **Test** | 181 | 1,006 | 10.0% |
| **Total** | 1,770 | 10,050 | 100.0% |

- **Split Seed**: `42`
- **Verification Integrity**:
  - Source-group overlap = 0
  - Exact duplicate overlap = 0
  - Known same-source near-duplicate overlap = 0
  - Missing files = 0
  - Invalid labels = 0
  - Corrupted images = 0

The split was kept strictly frozen throughout architecture benchmarking.

---

## 10.5.5 Class Definition

The task is four-class Wagner diabetic foot ulcer classification.

| Label | Class | Description |
|---:|---|---|
| 0 | **Grade 1** | Superficial ulcer not involving tendon, capsule, or bone |
| 1 | **Grade 2** | Deep ulcer involving tendon or capsule |
| 2 | **Grade 3** | Deep ulcer with abscess, osteomyelitis, or joint sepsis |
| 3 | **Grade 4** | Localized gangrene (forefoot or heel) |

Canonical class distribution:

| Class | Images | Percentage |
|---|---:|---:|
| **Grade 1** | 2,367 | 23.55% |
| **Grade 2** | 2,457 | 24.45% |
| **Grade 3** | 2,797 | 27.83% |
| **Grade 4** | 2,429 | 24.17% |

- **Class Imbalance Ratio**: 1.18 : 1 (reasonably balanced four-class dataset).

---

## 10.5.6 Frozen Benchmark Parameters

All candidate architectures were evaluated under identical experimental conditions.

| Configuration | Value |
|---|---|
| Input resolution | $224 \times 224$ |
| Channels | RGB (3) |
| Pretrained Weights | ImageNet pretrained |
| Head Dropout | 0.2 (uniform across candidates) |
| Random seed | 42 (Python, NumPy, PyTorch CPU/CUDA) |
| Batch size | 32 |
| Maximum epochs | 20 |
| Loss function | Unweighted Cross-Entropy |
| Optimizer | AdamW ($\text{lr} = 1\text{e-}4, \text{weight\_decay} = 1\text{e-}4$) |
| Scheduler | CosineAnnealingLR ($T_{\text{max}} = 20, \eta_{\text{min}} = 1\text{e-}6$) |
| Early stopping | Enabled (Patience = 10) |
| Checkpoint criterion | Minimum validation loss (`val_loss`) |
| Primary selection metric | Test Macro F1 |
| Execution Device | CUDA |
| Augmentation | Phase 10.2 approved Foot transforms |
| Test-set tuning | Not permitted |
