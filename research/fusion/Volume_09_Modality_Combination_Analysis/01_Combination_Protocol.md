# Phase C11.9 Research Protocol & Pre-Specified Questions

## 1. Scientific Context & Core Research Question

Phase C11.8 established the mechanics of decision authority reallocation and safety bounds when modalities are missing. Phase C11.9 addresses the distribution-level consequence of modality availability:

> **Does the frequency of a modality combination affect the behavior and reliability of ACARA-U? Does ACARA-U remain behaviorally robust across head and tail modality combinations?**

The conceptual motivation arises from real-world clinical deployment where multimodal completeness often exhibits a heavy-tailed distribution: complete tri-modal observations occur frequently in well-equipped specialized centers, whereas incomplete bimodal or unimodal observations represent long-tail clinical scenarios.

---

## 2. Seven Pre-Specified Research Questions

1. **RQ1 (Combination Routing Invariant)**: Does ACARA-U assign valid routing weights ($\sum_{i \in \mathcal{A}} w_i = 1.0, w_i \ge 0, w_{j \notin \mathcal{A}} = 0$) regardless of whether a modality combination belongs to the head, middle, or tail of the frequency distribution?
2. **RQ2 (Tail Risk Sensitivity)**: How does the decision-level risk sensitivity metric $D_{\text{tail}} = \mathbb{E}_{\text{tail}}[|R_{\text{fusion}} - R_{\text{fusion}}^{\text{RFC}}|]$ behave under moderate (D2) versus strong (D3) head-tail skew?
3. **RQ3 (Entropy & Authority Concentration)**: Does routing entropy $H(w)$ decrease systematically from head tri-modal packets to tail unimodal packets where authority necessarily collapses ($H=0$)?
4. **RQ4 (Uncertainty & DCRI Scaling)**: Does the additive uncertainty penalty $\delta U_{\text{sum}}$ in DCRI scale with available modality cardinality along nested channel ladders across all frequency tiers?
5. **RQ5 (Comparative Baseline Tail Robustness)**: How does ACARA-U compare against baseline fusion architectures (B1–B5) in tail risk shift $D_{\text{tail}}$ and dispersion $\sigma(R_{\text{tail}})$?
6. **RQ6 (Cross-Distribution Global Stability)**: Do population-level fused risk $\overline{R_{\text{fusion}}}$ and composite index $\overline{\text{DCRI}}$ remain stable across balanced (D1), moderate (D2), and strong-tail (D3) cohorts?
7. **RQ7 (Fail-Closed Safety)**: Is zero-modality fail-closed rejection ($\mathcal{A} = \emptyset \implies \text{status} = $ `NO_MODALITY_AVAILABLE`) preserved without distribution-dependent leakage?

---

## 3. Pre-Specified Scientific Hypotheses

- **Hypothesis H1 (Simplex & Safety Invariant Preservation across Tiers)**:  
  *Prediction*: Routing authority across active channels satisfies $\sum_{i \in \mathcal{A}} w_i = 1.0$ and $w_{j \notin \mathcal{A}} = 0.0$ uniformly across head, middle, and tail combinations under all distribution regimes.  
  *Status*: **Confirmed** (Mathematical invariant verified over 1,500 trials across D1, D2, D3).

- **Hypothesis H2 (Tail Risk Sensitivity Across Soft Baselines & Multi-Cohort Parity)**:  
  *Prediction*: ACARA-U achieves lower observed point-estimate tail risk sensitivity $D_{\text{tail}}$ than soft-allocation baselines (B2 Uniform Average, B3 Confidence-Weighted, B4 Conf+Rel, B5 Conf+Rel-Uncertainty) under long-tailed distributions (D2 and D3).  
  *Confirmatory Protocol ($S=30$ Cohorts, $N=15,000$ Packets)*: To resolve whether the single-cohort point-estimate difference between B6 and B5 is statistically distinguishable from zero, a pre-declared 30-cohort confirmatory protocol ($\text{seeds } 401\text{--}430$, $N=500$ each) was evaluated using a 2-stage hierarchical cluster bootstrap ($B=2,000$):
    - *Primary Endpoint*: D3 Strong Long-Tail paired difference $\Delta D_{\text{tail}} = D_{\text{tail}}(\text{B6}) - D_{\text{tail}}(\text{B5})$. Primary estimand is packet-weighted micro-average ($N_{\text{tail}}=35$ per cohort); sensitivity estimand is combination-weighted macro-average across $\{R, F, C\}$.
    - *Secondary Endpoint*: D2 Moderate Head-Tail paired difference $\Delta D_{\text{tail}}$ ($N_{\text{tail}}=75$ per cohort).
    - *Pre-Declared Decision Rules*:
      - *Practical Superiority*: Requires point estimate $\Delta < -0.005$, 95% hierarchical CI strictly $< 0$, and $p < 0.001$.
      - *Statistically Significant Modest Effect*: Requires $\Delta < 0$, 95% CI strictly $< 0$, and $p < 0.05$ (while failing practical superiority: $\Delta \ge -0.005$ or $p \ge 0.001$).
      - *Inferior*: Requires $\Delta \ge 0$ and 95% hierarchical CI strictly positive (lower bound $> 0$).
      - *Inconclusive (Not Statistically Distinguishable)*: If the 95% hierarchical CI contains zero or inferential criteria are discordant. An inconclusive finding means the data cannot distinguish between the models under the pre-declared rules; it does not prove statistical equivalence.
  *Status*: **Supported directionally in single cohort; Inconclusive (Not Statistically Distinguishable) in Multi-Cohort Confirmatory Evaluation**:
    - *Single-Cohort Reference ($\text{seed}=115$)*: $D_{\text{tail}}^{\text{ACARA-U}} = 0.1883$ vs $0.1889$ for B5 in D2; $0.1876$ vs $0.1889$ in D3 ($95\%$ paired CIs crossed zero).
    - *Multi-Cohort Confirmatory ($S=30$, $N=15,000$)*: In D3 (Primary), $\overline{D_{\text{tail}}}(\text{B6}) = 0.062143$ vs $\overline{D_{\text{tail}}}(\text{B5}) = 0.061962$ ($\Delta_{\text{micro}} = +0.000181$, $95\%$ Hierarchical CI: $[-0.000024, +0.000377]$, crossing zero; $\Delta_{\text{macro}} = +0.000123$, $95\%$ CI: $[-0.000093, +0.000344]$, crossing zero). In D2 (Secondary), $\overline{D_{\text{tail}}}(\text{B6}) = 0.065250$ vs $\overline{D_{\text{tail}}}(\text{B5}) = 0.065106$ ($\Delta_{\text{micro}} = +0.000143$, $95\%$ Hierarchical CI: $[-0.000025, +0.000300]$, crossing zero).
    - *Conclusion*: Under the unimodal tail tiers where single active modalities collapse router weights to $w_i \equiv 1.0$, the observed difference between B6 and B5 is not statistically distinguishable from zero under the hierarchical cluster bootstrap. Under the pre-declared decision protocol, the confirmatory evaluation concludes as `INCONCLUSIVE_NOT_STATISTICALLY_DISTINGUISHABLE`.

- **Hypothesis H3 (Tail Dispersion Expansion via Modality Collapse)**:  
  *Prediction*: Tail combinations exhibit increased risk standard deviation relative to the head ($\Delta \sigma(R) = \sigma(R_{\text{tail}}) - \sigma(R_{\text{head}}) > 0$), consistent with the reduced modality cardinality and resulting loss of multi-modal aggregation.  
  *Status*: **Supported descriptively** — Tail tiers exhibited higher observed risk dispersion than head tiers in both D2 ($\Delta \sigma = +0.0628$) and D3 ($\Delta \sigma = +0.0885$). The present analysis does not establish statistical significance of the dispersion difference.

---

## 4. Methodological Boundaries & Transparent Scope

- **Multi-Cohort Generative Simulation Boundary**: The 30 seeds generate independently sampled synthetic cohorts from the specified generative model; they represent controlled Monte Carlo simulation cohorts, not 30 independent clinical populations.
- **Controlled Decision-Level Cohort**: Because the underlying benchmark datasets (APTOS 2019 Retina, ADPM V3.3 Foot, UCI Diabetes Clinical) are retrospectively unpaired, C11.9 evaluates controlled decision packets ($N=500$, seed=115 for historical baseline; seeds 401–430 for confirmatory multi-cohort). It does not assert patient-level clinical prevalence for any combination.
- **Strictly Frozen Upstream Components**: Modality models, calibration maps, reliability constants ($R_R=0.929956, R_F=0.922266, R_C=0.825382$), router coefficients ($\alpha=1.0, \beta=1.5, \gamma=1.0, \eta=0.5$), and DCRI operating default ($\delta=0.20$, provisional evaluation default) remain strictly frozen.
- **No Retraining or Parameter Optimization**: C11.9 is an observational evaluation layer; no router retraining, Group-DRO, or adaptive MoE gating is introduced.
- **Participating Modality Probability Reporting**: Reports already-calibrated modality probability/confidence characteristics without fabricating a unified multimodal ground-truth label or computing fusion-level ECE.
- **No Label Leakage**: All combination-level metrics operate strictly on predictive channels without reference to ground-truth labels.
- **Volume 09 Sealed Status**: Phase C11.9 multi-cohort confirmatory verification complete with 8/8 gates passed and cryptographically sealed.
