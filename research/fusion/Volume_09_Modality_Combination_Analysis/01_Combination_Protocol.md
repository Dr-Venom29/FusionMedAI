# Phase C11.9 Research Protocol & Registered Questions

## 1. Scientific Context & Core Research Question

Phase C11.8 established the mechanics of decision authority reallocation and safety bounds when modalities are missing. Phase C11.9 addresses the distribution-level consequence of modality availability:

> **Does the frequency of a modality combination affect the behavior and reliability of ACARA-U? Does ACARA-U remain behaviorally robust across head and tail modality combinations?**

The conceptual motivation arises from real-world clinical deployment where multimodal completeness often exhibits a heavy-tailed distribution: complete tri-modal observations occur frequently in well-equipped specialized centers, whereas incomplete bimodal or unimodal observations represent long-tail clinical scenarios.

---

## 2. Seven Pre-Registered Research Questions

1. **RQ1 (Combination Routing Invariant)**: Does ACARA-U assign valid routing weights ($\sum_{i \in \mathcal{A}} w_i = 1.0, w_i \ge 0, w_{j \notin \mathcal{A}} = 0$) regardless of whether a modality combination belongs to the head, middle, or tail of the frequency distribution?
2. **RQ2 (Tail Risk Sensitivity)**: How does the decision-level risk sensitivity metric $D_{\text{tail}} = \mathbb{E}_{\text{tail}}[|R_{\text{fusion}} - R_{\text{fusion}}^{\text{RFC}}|]$ behave under moderate (D2) versus strong (D3) head-tail skew?
3. **RQ3 (Entropy & Authority Concentration)**: Does routing entropy $H(w)$ decrease systematically from head tri-modal packets to tail unimodal packets where authority necessarily collapses ($H=0$)?
4. **RQ4 (Uncertainty & DCRI Scaling)**: Does the additive uncertainty penalty $\delta U_{\text{sum}}$ in DCRI maintain monotonicity with respect to available modality cardinality across all frequency tiers?
5. **RQ5 (Comparative Baseline Tail Robustness)**: How does ACARA-U compare against baseline fusion architectures (B1–B5) in tail risk shift $D_{\text{tail}}$ and dispersion $\sigma(R_{\text{tail}})$?
6. **RQ6 (Cross-Distribution Global Stability)**: Do population-level fused risk $\overline{R_{\text{fusion}}}$ and composite index $\overline{\text{DCRI}}$ remain stable across balanced (D1), moderate (D2), and heavy-tailed (D3) cohorts?
7. **RQ7 (Fail-Closed Safety)**: Is zero-modality fail-closed rejection ($A = \emptyset \implies \text{NO\_MODALITY\_AVAILABLE}$) preserved without distribution-dependent leakage?

---

## 3. Registered Scientific Hypotheses

- **Hypothesis H1 (Simplex & Safety Invariant Preservation across Tiers)**:  
  *Prediction*: Routing authority across active channels satisfies $\sum_{i \in \mathcal{A}} w_i = 1.0$ and $w_{j \notin \mathcal{A}} = 0.0$ uniformly across head, middle, and tail combinations under all distribution regimes.  
  *Status*: **Confirmed** (Mathematical invariant verified over 1,500 trials across D1, D2, D3).

- **Hypothesis H2 (Tail Risk Sensitivity Across Soft Baselines)**:  
  *Prediction*: ACARA-U achieves lower observed point-estimate tail risk sensitivity $D_{\text{tail}}$ than soft-allocation baselines (B2 Uniform Average, B3 Confidence-Weighted, B4 Conf+Rel, B5 Conf+Rel-Uncertainty) under long-tailed distributions (D2 and D3).  
  *Interpretation Rule*: H2 is evaluated primarily as a point-estimate ordering hypothesis. Paired bootstrap confidence intervals are used to assess uncertainty around the B6-versus-B5 difference and are not interpreted as establishing superiority when they cross zero.  
  *Status*: **Supported** ($D_{\text{tail}}^{\text{ACARA-U}} = 0.1883$ vs $0.1925$ for B2, $0.1959$ for B3, $0.1967$ for B4, $0.1889$ for B5 under D2; $0.1876$ vs $0.1945$ for B2, $0.1974$ for B3, $0.1973$ for B4, $0.1889$ for B5 under D3. The margin over B5 is small: $+0.0006$ in D2 and $+0.0012$ in D3, with 95% paired bootstrap CIs crossing zero).

- **Hypothesis H3 (Tail Dispersion Expansion via Modality Collapse)**:  
  *Prediction*: Tail combinations exhibit increased risk standard deviation relative to the head ($\Delta \sigma(R) = \sigma(R_{\text{tail}}) - \sigma(R_{\text{head}}) > 0$), consistent with the reduced modality cardinality and resulting loss of multi-modal aggregation.  
  *Status*: **Supported descriptively** — Tail tiers exhibited higher observed risk dispersion than head tiers in both D2 ($\Delta \sigma = +0.0628$) and D3 ($\Delta \sigma = +0.0885$). The present analysis does not establish statistical significance of the dispersion difference.

---

## 4. Methodological Boundaries & Transparent Scope

- **Controlled Decision-Level Cohort**: Because the underlying benchmark datasets (APTOS 2019 Retina, ADPM V3.3 Foot, UCI Diabetes Clinical) are retrospectively unpaired, C11.9 evaluates controlled decision packets ($N=500$, seed=115). It does not assert patient-level clinical prevalence for any combination.
- **Strictly Frozen Upstream Components**: Modality models, calibration maps, reliability constants ($R_R=0.929956, R_F=0.922266, R_C=0.825382$), router coefficients ($\alpha=1.0, \beta=1.5, \gamma=1.0, \eta=0.5$), and DCRI operating default ($\delta=0.20$, provisional evaluation default) remain strictly frozen.
- **No Retraining or Parameter Optimization**: C11.9 is an observational evaluation layer; no router retraining, Group-DRO, or adaptive MoE gating is introduced.
- **Participating Modality Probability Reporting**: Reports already-calibrated modality probability/confidence characteristics without fabricating a unified multimodal ground-truth label or computing fusion-level ECE.
- **No Label Leakage**: All combination-level metrics operate strictly on predictive channels without reference to ground-truth labels.
