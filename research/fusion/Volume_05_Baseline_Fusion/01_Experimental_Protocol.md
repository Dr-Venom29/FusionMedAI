# Research Protocol: Multimodal Baseline Evaluation

## Protocol Governance & Scope Declaration

This document establishes the experimental protocol for evaluating decision-level fusion baselines in Phase C11.5 of **FusionMedAI**.

### 1. Scientific Boundaries & Prohibited Practices

To preserve methodological integrity across independently developed modalities, Phase C11.5 enforces the following invariant constraints:

1. **Zero Fake Patient Pairing**: The validation prediction pools for Retina ($N=366$ full val images), Foot ($N=1006$ full val images), and Clinical ($N=1066$ representative validation records sampled via stride-14 across the $14,911$ validation encounters) originate from independent clinical distributions (APTOS 2019, DFUC 2020 / Wagner, and CDC Diabetes Health Indicators). Phase C11.5 evaluates baselines strictly over `CONTROLLED_DECISION_PACKET` structures. It does **not** construct synthetic cross-modality "virtual patients" or claim empirical joint diagnosis.
2. **Zero Ground-Truth Label Contamination**: The dynamic routing kernel and baseline weighting rules operate strictly on prediction signals: instance risk $r_i$, confidence $C_i$, uncertainty $U_i$, input quality $Q_i$, and historical reliability priors $R_i$. Ground-truth labels ($y_{\text{true}}$) are strictly prohibited from baseline routing signatures.
3. **Decision-Level Scope**: Baselines compute an aggregated decision-level risk index:
   $$R_{\text{fusion}} = \sum_{i \in \mathcal{A}} w_i r_i$$
   This index represents a composite risk summary across available sub-networks and is **not** calibrated as a true patient joint posterior probability.

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

## Frozen Historical Validation Priors

All baselines incorporating historical reliability ($R_i$) consume the exact constants frozen in Phase C11.3:

$$\begin{aligned}
R_{\text{Retina}} &= 0.929956 \quad (\text{AUC}=0.911149, \text{ECE}=0.051237) \\
R_{\text{Foot}}   &= 0.922266 \quad (\text{AUC}=0.884435, \text{ECE}=0.039903) \\
R_{\text{Clinical}} &= 0.825382 \quad (\text{AUC}=0.650764, \text{ECE}=0.000000)
\end{aligned}$$

No retraining, dynamic recalibration, or online tuning of these priors is permitted during baseline evaluation.
