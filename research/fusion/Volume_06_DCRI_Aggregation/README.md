# Volume 06: DCRI Risk Aggregation & Uncertainty Discounting

**Phase C11.6 — Research Compendium & Technical Specifications**  
*FusionMedAI Multimodal Fusion Engine*

---

## Executive Overview

Volume 06 establishes the mathematical and behavioral properties of the **Decision-Critical Risk Index (DCRI)** aggregation layer (`DCRIEngine` and `ResearchFusionIndex`). Operating downstream of `RetinaAdapter`, `FootAdapter`, and `ClinicalAdapter` $\to$ `EvidenceOrchestrator` $\to$ `ACARAUv2Router`, Phase C11.6 preserves native modality predictions inside `EvidenceReport` (`common_target_prediction = None`) while converting modality authority weights $w_i$, scalar risk projections $r_i$, and predictive uncertainties $U_i$ into a continuous research fusion index $R_{\text{fusion}}$ and an uncertainty-discounted benchmark metric $\text{DCRI}_\delta$ inside `ResearchFusionIndex(is_calibrated_clinical_probability=False, authorized_for_clinical_action=False)`.

### Core Mathematical Equations

$$
\boxed{R_{\text{fusion}} = \sum_{i \in \mathcal{A}} w_i r_i = \sum_{i \in \mathcal{A}} K_i}
$$

$$
\boxed{U_{\text{sum}} = \sum_{i \in \mathcal{A}} U_i, \quad U_{\text{mean}} = \frac{1}{\lvert \mathcal{A} \rvert} \sum_{i \in \mathcal{A}} U_i}
$$

$$
\boxed{P_U(\delta) = \delta \sum_{i \in \mathcal{A}} U_i = \delta U_{\text{sum}}}
$$

$$
\boxed{\text{DCRI}_\delta = R_{\text{fusion}} - P_U(\delta) = \sum_{i \in \mathcal{A}} w_i r_i - \delta \sum_{i \in \mathcal{A}} U_i}
$$

```text
   Modality Adapters (RetinaAdapter / FootAdapter / ClinicalAdapter)
                             │
                             ▼
            ┌─────────────────────────────────┐
            │ EvidenceOrchestrator (N=500)    │
            │  • EvidenceReport (Native Preds)│
            │  • ACARAUv2Router (weights wi)  │
            └────────────────┬────────────────┘
                             │
                             ▼
                        weights wi
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
          risks ri                    uncertainties Ui
              │                             │
              ▼                             ▼
       wi × ri = Ki                   Σ Ui = U_sum
              │                             │
              └──────────────┬──────────────┘
                             ▼
                     R_fusion = Σ Ki
                             │
                             │   δ ∈ {0, 0.05, 0.1, 0.2, 0.5, 1.0}
                             ▼
                    uncertainty penalty
                        P_U = δ Σ Ui
                             │
                             ▼
            ResearchFusionIndex: DCRI = R_fusion - P_U
```

---

## Volume 06 Document Index

| Document | Focus & Content |
| :--- | :--- |
| [01_DCRI_Protocol.md](01_DCRI_Protocol.md) | Scientific objectives, input-output contracts, and methodological boundaries. |
| [02_Mathematical_Formulation.md](02_Mathematical_Formulation.md) | Formal derivation, bounds $[-\delta M, 1]$, and non-clamping policy. |
| [03_Aggregation_Procedure.md](03_Aggregation_Procedure.md) | Weighted risk contribution $K_i = w_i r_i$ and aggregation mechanics. |
| [04_Delta_Sensitivity.md](04_Delta_Sensitivity.md) | Sensitivity grid evaluation across $\delta \in \{0, 0.05, 0.1, 0.2, 0.5, 1.0\}$. |
| [05_Uncertainty_Penalty_Analysis.md](05_Uncertainty_Penalty_Analysis.md) | Additive burden scaling vs mean uncertainty and double-use analysis. |
| [06_Modality_Contribution_Analysis.md](06_Modality_Contribution_Analysis.md) | Modality authority vs risk contribution shares across 7 availability regimes. |
| [07_Results.md](07_Results.md) | Empirical statistical results over the $N=500$ controlled cohort. |
| [08_Freeze_Report.md](08_Freeze_Report.md) | Phase C11.6 certification, verification gate summary (16/16), and artifact freeze. |

---

## Key Findings & Boundaries

1. **Non-Clamping Policy**: DCRI is strictly maintained in $[-\delta M, 1]$ without artificial truncation to $[0, 1]$. In the $N=500$ cohort, $\overline{\mathrm{DCRI}}_{0.10} = 0.196229$ ($39$ negative packets, $7.8\%$ under $\Theta_{\text{unit}}$) and $\overline{\mathrm{DCRI}}_{0.10} = 0.197624$ ($38$ negative packets, $7.6\%$ under $\Theta_0$), while at $\delta=0.20$ $\overline{\mathrm{DCRI}}_{0.20} = 0.132835$ ($128$ negative packets, $25.6\%$ under $\Theta_{\text{unit}}$) and $0.134230$ ($127$ negative packets, $25.4\%$ under $\Theta_0$).
2. **5-Map Inactive Zeroing Invariant**: `DCRIEngine.evaluate_packet()` and `DCRIResult.__post_init__` enforce exact $0.0$ entries across all 5 channel maps (`modality_weights`, `modality_risks`, `modality_uncertainties`, `weighted_risk_contributions`, `uncertainty_penalty_contributions`) for inactive modalities.
3. **Double-Use of Uncertainty**: Uncertainty modulates routing weights $w_i$ (relative authority modulation) and applies an absolute additive discount $P_U$ (epistemic risk reduction).
4. **Delta Sensitivity**: Empirical shift exactly matches theoretical sensitivity slope $\frac{\partial \overline{\text{DCRI}}}{\partial \delta} = -\overline{U_{\text{sum}}} = -0.633936$.
5. **No Premature Delta Selection**: Final parameter optimization is deferred to Phase C11.13.

