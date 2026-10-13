# Volume 02 — Unified Input Quality & Availability Layer (Phase C11.2)

## Volume Overview & Scope

This volume contains the scientific and technical specifications for the **FusionMedAI Quality & Availability Ingestion Layer** (Phase C11.2).

The quality layer enforces strict separation between input fidelity and downstream predictive inference. It transforms raw, unprocessed inputs from all three clinical modalities into deterministic $(A_i, Q_i)$ tuples that are ingested by `RetinaAdapter`, `FootAdapter`, and `ClinicalAdapter` into `EvidenceOrchestrator` before entering the ACARA-U v2 routing kernel.

---

## Documents in this Volume

| Document | Primary Focus | Key Invariants Frozen |
| :--- | :--- | :--- |
| [`01_Quality_and_Availability_Protocol.md`](./01_Quality_and_Availability_Protocol.md) | Quality & Availability Governance | $A_i \in \{0, 1\}$, $Q_i \in [0, 1]$, $A_i=0 \implies Q_i=0$, zero label/prediction leakage, frozen training bounds |

---

## Modality Quality Summary

- **Retina ($Q_R$)**: $\frac{1}{2}(Q_{\text{sharp}} + Q_{\text{illum}})$ via Laplacian variance and dynamic range luminance penalties (`RetinaAdapter`, cohort mean $\bar{Q}_R = 0.768986$).
- **Foot ($Q_F$)**: $\frac{1}{2}(Q_{\text{CNR}} + Q_{\text{boundary}})$ via unsupervised Otsu contrast and Sobel gradient-magnitude boundary clarity (`FootAdapter`, cohort mean $\bar{Q}_F = 0.708265$).
- **Clinical ($Q_C$)**: Feature completeness ratio $\frac{N_{\text{valid}}}{119}$ evaluated after transformation through the frozen 119-D `ClinicalPreprocessor` (`ClinicalAdapter`, cohort mean $\bar{Q}_C = 0.858804$).
- **Unmeasured Quality Fallback**: When raw quality is unmeasured (`quality=None`), the modality adapters assign a neutral $Q_i = 0.5$ fallback and explicitly record `"quality_is_measured": False` in `ModalityPrediction` provenance metadata.

