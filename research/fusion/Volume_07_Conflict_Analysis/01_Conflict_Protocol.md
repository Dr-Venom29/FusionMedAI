# Phase C11.7 Research Protocol & Registered Questions

## 1. Scientific Context & Core Research Question

Phase C11.7 evaluates cross-modality discordance at the decision output layer of the multimodal pipeline. The primary research question is:

> **When available prediction modalities disagree on projected risk, can the system detect, quantify, and attribute the conflict without falsely interpreting disagreement as multimodal synergy or clinical certainty?**

---

## 2. Seven Pre-Specified Research Questions

1. **RQ1 (Conflict Detection)**: Can the metric family reliably identify decision packets exhibiting substantial inter-channel risk divergence?
2. **RQ2 (Availability Sensitivity)**: How do conflict metrics behave across unimodal, bimodal, and tri-modal availability regimes?
3. **RQ3 (Weight-Conflict Dynamics)**: When conflict is high, does ACARA-U concentrate authority on high-confidence channels or distribute it across modalities?
4. **RQ4 (Uncertainty-Conflict Orthogonality)**: Is inter-channel risk divergence statistically distinct from single-modality epistemic/aleatoric uncertainty?
5. **RQ5 (Reliability Prior Alignment)**: Does ACARA-U systematically assign greater authority to channels with higher global reliability priors ($R_i$) under high-conflict conditions?
6. **RQ6 (Perturbation Monotonicity)**: Do the conflict metrics respond predictably and monotonically in the expected direction under controlled outward and inward risk perturbations?
7. **RQ7 (Structured Diagnostic Explainability)**: Can every decision packet produce deterministic, structured diagnostic explanations identifying conflicting pairs, authority allocations, and uncertainty context?

---

## 3. Explicit Methodological Boundaries

- **No Patient-Level Ground Truth**: Retrospective imaging and EHR datasets are unpaired; evaluations benchmark decision-level consistency.
- **No Weight / Threshold Optimization**: Conflict metrics are diagnostic. No parameters are fitted or adjusted to reduce conflict scores.
- **Zero Upstream Mutation**: Upstream router weights ($w_i$), fused risk ($R_{\text{fusion}}$), and uncertainty-discounted index ($\text{DCRI}_\delta$) remain strictly immutable.
- **Ground-Truth Independence**: Zero class labels or outcome targets are consumed during conflict computation.
