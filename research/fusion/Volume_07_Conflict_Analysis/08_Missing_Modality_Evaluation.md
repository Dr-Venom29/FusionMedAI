# Modality Availability & Missingness Analysis

## 1. Multi-Configuration Performance Matrix ($N=500$)

Across the frozen $N=500$ cohort, conflict metrics were systematically evaluated across all 7 availability configurations plus zero-modality:

| Availability Configuration | Active Count ($M$) | Conflict Available | Mean $\Delta_{\max}$ | Mean $\Delta_{\text{mean}}$ | Mean $\sigma_w$ | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Retina Only (R)** | 1 | False | None | None | 0.000000 | `NOT_APPLICABLE` |
| **Foot Only (F)** | 1 | False | None | None | 0.000000 | `NOT_APPLICABLE` |
| **Clinical Only (C)** | 1 | False | None | None | 0.000000 | `NOT_APPLICABLE` |
| **Retina + Foot (RF)** | 2 | True | 0.391578 | 0.391578 | 0.177265 | `SUCCESS` |
| **Retina + Clinical (RC)** | 2 | True | 0.234970 | 0.234970 | 0.100913 | `SUCCESS` |
| **Foot + Clinical (FC)** | 2 | True | 0.414113 | 0.414113 | 0.198308 | `SUCCESS` |
| **Tri-Modal (RFC)** | 3 | True | **0.520331** | **0.346887** | **0.215019** | `SUCCESS` |
| **Zero Modality** | 0 | False | None | None | 0.000000 | `NO_MODALITY_AVAILABLE` |

---

## 2. Invariants Across Availability Regimes

1. **Unimodal Handling**: In single-modality settings ($M=1$), pairwise disagreement is formally undefined (`None`) and `conflict_available = False`, preventing the false conflation of "no disagreement" with "disagreement cannot be measured."
2. **Bimodal Equivalence**: In two-modality settings ($M=2$), $\Delta_{\max} \equiv \Delta_{\text{mean}} \equiv |r_j - r_k|$, as exactly one pair exists ($P=1$).
3. **Tri-Modal Expansion**: In tri-modal settings ($M=3$), $\Delta_{\max}$ increases to $0.5203$ because it captures the extreme envelope of all 3 channels, while $\Delta_{\text{mean}} = 0.3469$ reflects average pairwise tension.
4. **Zero-Modality Safe Failure**: When no channels are available, `NO_MODALITY_AVAILABLE` is emitted without raising uncaught exceptions.
