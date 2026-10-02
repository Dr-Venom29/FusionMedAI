# Volume 02 — Unified Input Quality & Availability Layer (Phase C11.2)

## Volume Overview & Scope

This volume contains the scientific and technical specifications for the **FusionMedAI Quality & Availability Ingestion Layer** (Phase C11.2).

The quality layer enforces strict separation between input fidelity and downstream predictive inference. It transforms raw, unprocessed inputs from all three clinical modalities into deterministic $(A_i, Q_i)$ tuples before entering the ACARA-U decision fusion architecture.

---

## Documents in this Volume

| Document | Primary Focus | Key Invariants Frozen |
| :--- | :--- | :--- |
| [`01_Quality_and_Availability_Protocol.md`](./01_Quality_and_Availability_Protocol.md) | Quality & Availability Governance | $A_i \in \{0, 1\}$, $Q_i \in [0, 1]$, $A_i=0 \implies Q_i=0$, zero label/prediction leakage, frozen training bounds |

---

## Modality Quality Summary

- **Retina ($Q_R$)**: $\frac{1}{2}(Q_{\text{sharp}} + Q_{\text{illum}})$ via Laplacian variance and dynamic range luminance penalties.
- **Foot ($Q_F$)**: $\frac{1}{2}(Q_{\text{CNR}} + Q_{\text{boundary}})$ via unsupervised Otsu contrast and Sobel gradient-magnitude boundary clarity.
- **Clinical ($Q_C$)**: Feature completeness ratio $\frac{N_{\text{valid}}}{119}$ (raw encounters are transformed through the frozen 119-D `ClinicalPreprocessor` before completeness evaluation).
