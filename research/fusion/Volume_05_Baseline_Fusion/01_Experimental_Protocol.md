# Research Protocol: Multimodal Baseline Evaluation

## Protocol Governance & Scope Declaration

This document establishes the experimental protocol for evaluating decision-level fusion baselines in Phase C11.5 of **FusionMedAI**.

> [!IMPORTANT]
> **Scientific Scope & Dual-Hypothesis Separation**:
> Phase C11.5 deliberately separates two scientific claims:
> 1. **Claim 1 (Evaluated in C11.5)**: ACARA-U behaves strictly according to its mathematical design, demonstrating predictable, directional, and monotonic authority allocation across varying confidence, uncertainty, input quality, reliability, and missing-modality states under controlled evaluation.
> 2. **Claim 2 (Reserved for Future Paired-Cohort Studies)**: Does multimodal routing improve patient-level diagnostic accuracy (AUROC, AUPRC, Brier score, ECE) relative to unimodal or heuristic baselines when all modalities observe the same patient? This claim requires a unified, genuinely paired multimodal cohort with a common clinical ground truth ($y_{\text{patient}}$) and cannot be answered using independent unlinked datasets.
>
> **Core Research Question Addressed in Phase C11.5**:
> *"Does uncertainty-, reliability-, quality-, and availability-aware decision-level routing provide more robust, predictable, and safety-compliant modality authority allocation under controlled incomplete multimodal conditions?"*

### 1. Scientific Boundaries & Prohibited Practices

To preserve methodological integrity across independently developed modalities, Phase C11.5 enforces the following invariant constraints:

1. **Zero Fake Patient Pairing**: The validation prediction pools for Retina ($N=366$ full val images), Foot ($N=1006$ full val images), and Clinical ($N=1066$ representative validation records sampled via stride-14 across the $14,911$ validation encounters) originate from independent clinical distributions (APTOS 2019, DFUC 2020 / Wagner, and CDC Diabetes Health Indicators). Phase C11.5 evaluates baselines strictly over `CONTROLLED_DECISION_PACKET` structures. It does **not** construct synthetic cross-modality "virtual patients" or claim empirical joint diagnosis.
2. **Zero Ground-Truth Label Contamination & No Synthetic Targets**: The dynamic routing kernel and baseline weighting rules operate strictly on prediction signals: instance risk $r_i$, confidence $C_i$, uncertainty $U_i$, input quality $Q_i$, and historical reliability priors $R_i$. Ground-truth labels ($y_{\text{true}}$) are strictly prohibited from baseline routing signatures. Furthermore, constructing synthetic pseudo-targets (e.g., $y_{\text{synthetic}} = \sum \alpha_i y_i$) to manufacture AUROC/ECE metrics is strictly prohibited, as this would evaluate whether the fusion recovers an arbitrary synthetic formula rather than validating real clinical predictive performance.
3. **Decision-Level Scope**: Baselines compute an aggregated decision-level risk index:
   $$R_{\text{fusion}} = \sum_{i \in \mathcal{A}} w_i r_i$$
   This index represents a composite risk summary across available sub-networks and is **not** calibrated as a true patient joint posterior probability. Evaluation analyzes behavioral properties (modality contribution $w_i r_i$, cross-modality divergence $\Delta_{\text{conflict}}$, routing entropy $H(w)$, and perturbation sensitivities) rather than clinical accuracy.

---

## Two-Tier Evaluation Architecture

Phase C11.5 adheres to the two-tier evaluation framework formalized in Phase C11.0:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        Phase C11.5 Evaluation                          │
├────────────────────────────────────────────────────────────────────────┤
│  Tier 1: Behavioral & Sensitivity Evaluation                           │
│  ├── Weight allocation dynamics across B1–B6                           │
│  ├── Routing entropy: H(w) = -Σ w_i ln(w_i)                            │
│  ├── Dominant modality distribution                                    │
│  ├── Step-response curves (Confidence, Uncertainty, Quality sweeps)    │
│  └── Cross-modality risk divergence (X_RF, X_RC, X_FC, X_max)          │
├────────────────────────────────────────────────────────────────────────┤
│  Tier 2: Robustness & Missing-Modality Matrix                          │
│  ├── 7 operational availability configurations (C1 to C7)              │
│  ├── Graceful degradation under single and dual modality dropouts      │
│  └── Zero-modality safe rejection (Config 0: A_R=A_F=A_C=0)            │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Baseline Ladder Structure (B1–B6)

The baseline ladder isolates each informational component systematically:

| ID | Baseline Name | Scoring Rule / Decision Formulation | Target Informational Role |
| :--- | :--- | :--- | :--- |
| **B1** | Reliability-Selected | $w_{i^*} = 1.0 \text{ where } i^* = \arg\max_{i \in \mathcal{A}} R_i$ | Static prior-only unimodal upper-bound baseline |
| **B2** | Uniform Average | $w_i = 1 / |\mathcal{A}|$ | Unweighted consensus fusion baseline |
| **B3** | Confidence-Only | $z_i = 1.0 \times C_i$ | Dynamic prediction strength baseline |
| **B4** | Confidence + Reliability | $z_i = 1.0 \times C_i + 1.0 \times R_i$ | Dynamic strength + frozen historical validation prior |
| **B5** | Conf + Rel + Uncertainty | $z_i = 1.0 \times C_i + 1.0 \times R_i - 1.0 \times U_i$ | Uncertainty-penalized dynamic routing |
| **B6** | Full ACARA-U | $z_i = 1.0 \times C_i + 1.0 \times R_i - 1.0 \times U_i + 1.0 \times Q_i$ | Full dynamic router (C11.4 reference wrapper) |

---

## Modality Availability Matrix Configurations

Baselines are evaluated across all $2^3 = 8$ availability states:

- **Config 1 (Tri-Modal)**: $\mathcal{A} = \{\text{Retina}, \text{Foot}, \text{Clinical}\}$ ($A_R=1, A_F=1, A_C=1$)
- **Config 2 (Dual-Modal RF)**: $\mathcal{A} = \{\text{Retina}, \text{Foot}\}$ ($A_R=1, A_F=1, A_C=0$)
- **Config 3 (Dual-Modal RC)**: $\mathcal{A} = \{\text{Retina}, \text{Clinical}\}$ ($A_R=1, A_F=0, A_C=1$)
- **Config 4 (Dual-Modal FC)**: $\mathcal{A} = \{\text{Foot}, \text{Clinical}\}$ ($A_R=0, A_F=1, A_C=1$)
- **Config 5 (Unimodal R)**: $\mathcal{A} = \{\text{Retina}\}$ ($A_R=1, A_F=0, A_C=0$)
- **Config 6 (Unimodal F)**: $\mathcal{A} = \{\text{Foot}\}$ ($A_R=0, A_F=1, A_C=0$)
- **Config 7 (Unimodal C)**: $\mathcal{A} = \{\text{Clinical}\}$ ($A_R=0, A_F=0, A_C=1$)
- **Config 0 (Zero Available)**: $\mathcal{A} = \emptyset$ ($A_R=0, A_F=0, A_C=0$) $\implies$ Safe rejection (`NO_MODALITY_AVAILABLE`)

---

## Frozen Historical Validation Priors

All baselines incorporating historical reliability ($R_i$) consume the exact constants frozen in Phase C11.3:

$$\begin{aligned}
R_{\text{Retina}} &= 0.929956 \quad (\text{AUC}=0.911149, \text{ECE}=0.051237) \\
R_{\text{Foot}}   &= 0.922266 \quad (\text{AUC}=0.884435, \text{ECE}=0.039903) \\
R_{\text{Clinical}} &= 0.825382 \quad (\text{AUC}=0.650764, \text{ECE}=0.000000)
\end{aligned}$$

No retraining, dynamic recalibration, or online tuning of these priors is permitted during baseline evaluation.

