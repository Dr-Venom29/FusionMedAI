# Phase C11.8 Research Protocol & Registered Questions

## 1. Scientific Context & Core Research Question

Phase C11.8 evaluates the robustness of multimodal decision fusion under missing modality availability. The primary research question is:

> **When one or more modalities are unavailable, does ACARA-U degrade gracefully by reallocating decision authority among the remaining modalities, while avoiding invalid fusion, unstable routing, or misleading confidence?**

---

## 2. Seven Pre-Specified Research Questions

1. **RQ1 (Availability Safety)**: Does the router assign exactly zero authority ($w_i = 0.0$) to unavailable modalities under all conditions?
2. **RQ2 (Authority Redistribution)**: How does ACARA-U redistribute decision authority among the remaining active channels when 1 or 2 modalities disappear?
3. **RQ3 (Simplex Renormalization)**: Is $\sum_{i \in \mathcal{A}} w_i = 1.0$ and $w_i \in [0.0, 1.0]$ strictly preserved across all 7 non-empty availability states?
4. **RQ4 (Risk Continuity & Sensitivity)**: What is the magnitude and distribution of $\Delta R_{\text{missing}} = |R_{\text{fusion}}^{\text{full}} - R_{\text{fusion}}^{\text{subset}}|$ across dropout configurations?
5. **RQ5 (Uncertainty & DCRI Response)**: How do $U_{\text{sum}}$ and $\text{DCRI}_\delta$ respond under modality loss, and how does the signed decomposition $\Delta \text{DCRI} = \Delta R_{\text{fusion}} - \delta \Delta U_{\text{sum}}$ behave?
6. **RQ6 (Graceful Rejection)**: Does the system fail safely when $\mathcal{A} = \emptyset$, returning status `NO_MODALITY_AVAILABLE` with zero numerical outputs?
7. **RQ7 (Comparative Baseline Behavior)**: How does ACARA-U's authority redistribution and decision-level risk sensitivity compare with baseline routing strategies (B1–B5) under controlled modality loss?

---

## 3. Pre-Specified Scientific Hypotheses

- **Hypothesis H1 (Availability Safety & Simplex Conservation)**:  
  ACARA-U will satisfy strict availability safety under all modality-loss configurations ($A_i = 0 \implies w_i = 0.0$) and preserve normalized authority ($\sum_{i \in \mathcal{A}} w_i = 1.0$) across all non-empty subsets.
- **Hypothesis H2 (Comparative Missing-Modality Sensitivity)**:  
  ACARA-U exhibits distinct and empirically bounded decision-level risk sensitivity under controlled modality loss compared with the evaluated baseline strategies.
- **Hypothesis H3 (Uncertainty-Stratified Missingness Sensitivity)**:  
  The decision-level impact of modality removal differs between low- and high-uncertainty strata, indicating an association between instance-level predictive uncertainty and missing-modality sensitivity.

---

## 4. Methodological Boundaries

- **Controlled Decision-Level Scope**: C11.8 evaluates robustness of the fusion mechanism to controlled availability changes; it does not establish that a missing modality causes a clinically meaningful deterioration in patient risk prediction.
- **No Retraining or Parameter Tuning**: Router coefficients ($\alpha=1.0, \beta=1.5, \gamma=1.0, \eta=0.5$), reliability priors, and $\delta=0.20$ evaluation point remain strictly frozen.
- **Zero Label Leakage**: Ground-truth labels are completely decoupled from decision-level missingness evaluations.
- **No Generative Imputation**: Feature-level imputation or hallucination is prohibited; missingness is resolved strictly at the decision authority layer.
