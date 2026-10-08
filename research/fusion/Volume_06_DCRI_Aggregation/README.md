# Volume 06: DCRI Risk Aggregation & Uncertainty Discounting

**Phase C11.6 — Research Compendium & Technical Specifications**  
*FusionMedAI Multimodal Fusion Engine*

---

## Executive Overview

Volume 06 establishes the mathematical and behavioral properties of the **Decision-Critical Risk Index (DCRI)** aggregation layer. Operating downstream of the ACARA-U dynamic routing engine (Phase C11.4 / C11.5), Phase C11.6 converts modality authority weights $w_i$, scalar risk projections $r_i$, and predictive uncertainties $U_i$ into a continuous fused risk index $R_{\text{fusion}}$ and an uncertainty-discounted decision metric $\text{DCRI}_\delta$.

### Core Mathematical Equations

$$\boxed{R_{\text{fusion}} = \sum_{i \in \mathcal{A}} w_i r_i = \sum_{i \in \mathcal{A}} K_i}$$

$$\boxed{U_{\text{sum}} = \sum_{i \in \mathcal{A}} U_i, \quad U_{\text{mean}} = \frac{1}{|\mathcal{A}|} \sum_{i \in \mathcal{A}} U_i}$$

$$\boxed{P_U(\delta) = \delta \sum_{i \in \mathcal{A}} U_i = \delta U_{\text{sum}}}$$

$$\boxed{\text{DCRI}_\delta = R_{\text{fusion}} - P_U(\delta) = \sum_{i \in \mathcal{A}} w_i r_i - \delta \sum_{i \in \mathcal{A}} U_i}$$

```text
               C11.5 Frozen Packets (N=500, seed=115)
                             │
                             ▼
                    ┌─────────────────┐
                    │ ACARA-U Router  │
                    └────────┬────────┘
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
                    DCRI = R_fusion - P_U
```

---

## Volume 06 Document Index

| Document | Focus & Content |
| :--- | :--- |
| [01_DCRI_Protocol.md](file:///D:/FusionMedAI/research/fusion/Volume_06_DCRI_Aggregation/01_DCRI_Protocol.md) | Scientific objectives, input-output contracts, and methodological boundaries. |
| [02_Mathematical_Formulation.md](file:///D:/FusionMedAI/research/fusion/Volume_06_DCRI_Aggregation/02_Mathematical_Formulation.md) | Formal derivation, bounds $[-\delta M, 1]$, and non-clamping policy. |
| [03_Aggregation_Procedure.md](file:///D:/FusionMedAI/research/fusion/Volume_06_DCRI_Aggregation/03_Aggregation_Procedure.md) | Weighted risk contribution $K_i = w_i r_i$ and aggregation mechanics. |
| [04_Delta_Sensitivity.md](file:///D:/FusionMedAI/research/fusion/Volume_06_DCRI_Aggregation/04_Delta_Sensitivity.md) | Sensitivity grid evaluation across $\delta \in \{0, 0.05, 0.1, 0.2, 0.5, 1.0\}$. |
| [05_Uncertainty_Penalty_Analysis.md](file:///D:/FusionMedAI/research/fusion/Volume_06_DCRI_Aggregation/05_Uncertainty_Penalty_Analysis.md) | Additive burden scaling vs mean uncertainty and double-use analysis. |
| [06_Modality_Contribution_Analysis.md](file:///D:/FusionMedAI/research/fusion/Volume_06_DCRI_Aggregation/06_Modality_Contribution_Analysis.md) | Modality authority vs risk contribution shares across 7 availability regimes. |
| [07_Results.md](file:///D:/FusionMedAI/research/fusion/Volume_06_DCRI_Aggregation/07_Results.md) | Empirical statistical results over the $N=500$ controlled cohort. |
| [08_Freeze_Report.md](file:///D:/FusionMedAI/research/fusion/Volume_06_DCRI_Aggregation/08_Freeze_Report.md) | Phase C11.6 certification, verification gate summary (16/16), and artifact freeze. |

---

## Key Findings & Boundaries

1. **Non-Clamping Policy**: DCRI is strictly maintained in $[-\delta M, 1]$ without artificial truncation to $[0, 1]$. In the frozen $N=500$ cohort, the observed minimum at $\delta=0.20$ was $-0.125243$ ($24.6\%$ negative rate) due to low baseline risk and high uncertainty burden.
2. **Double-Use of Uncertainty**: Uncertainty modulates routing weights $w_i$ (relative authority modulation) and applies an absolute additive discount $P_U$ (epistemic risk reduction).
3. **Delta Sensitivity**: Empirical shift exactly matches theoretical sensitivity slope $\frac{\partial \overline{\text{DCRI}}}{\partial \delta} = -\overline{U_{\text{sum}}} = -0.633936$.
4. **No Premature Delta Selection**: Final parameter optimization is deferred to Phase C11.12.
