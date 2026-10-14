# Phase C11.18: Failure & Error Analysis Protocol

## 1. Executive Protocol Overview
- **Phase Code**: C11.18
- **Phase Title**: Detailed Failure & Error Analysis Procedure
- **Parent Evaluation Phase**: C11.17 (Final Decision-Level Evaluation of ACARA-U)
- **Subsequent Phase**: C11.19 (Final Research Freeze & Deployment Readiness)
- **Primary Objective**: Investigate, quantify, categorize, and explain operational failure modes, boundary edge conditions, and decision anomalies exposed during the frozen C11.17 decision evaluation.
- **Operating Boundary**: Strict post-hoc forensic analysis. **No router retuning, coefficient adjustments, threshold recalibrations, or packet re-syntheses are permitted.**

---

## 2. Frozen Reference Configuration & Modality Scope
The parameters governing the evaluation cohort are strictly frozen from Phase C11.17:
- **Router Coefficients**: $\alpha = 1.0$ (predictive confidence), $\beta = 1.5$ (historical reliability prior), $\gamma = 1.0$ (epistemic uncertainty penalty), $\eta = 0.5$ (input quality fidelity).
- **DCRI Definition & Active Channel Scope**:
  $$\text{DCRI}_{0.10} = R_{\mathrm{fusion}} - 0.10 \sum_{i \in \mathcal{A}} U_i$$
  In the frozen C11.17 implementation, $\mathcal{A}$ denotes the set of all available modality sensors present in the decision packet ($|\mathcal{A}| = 3$ for the complete-case cohort: Retina, Foot, Clinical).
  - For **B2–B6**, all available modalities enter the routing kernel and receive non-zero weights.
  - For **B1 (Reliability-Selected Unimodal)**, $R_{\mathrm{fusion}} = r_{\mathrm{retina}}$ (since $w_R = 1.0, w_F = 0, w_C = 0$), but the linear uncertainty penalty subtracts cumulative uncertainty across all three available channels ($U_R + U_F + U_C$). This structural formulation-policy mismatch (single-channel risk minus multi-channel uncertainty) is a key factor in B1's negative score rate.
- **Negative Clamping Policy**: `false` (unclamped DCRI scores preserved for mathematical transparency and auditing).
- **Decision Tier Thresholds**:
  - $\tau_1 = 0.20$: Threshold separating Routine Review (Tier 0) from Additional Assessment (Tier 1).
  - $\tau_2 = 0.40$: Threshold separating Additional Assessment (Tier 1) from Escalation for Human Review (Tier 2).
- **Primary Cohort Scope**: Sealed Phase C11.17 baseline evaluation cohort ($N = 500$ controlled decision packets, generation seed = $115$, raw probability confidence representation).
- **Secondary Cross-Audit Scope**: Canonical redesigned rerun cohort ($N = 500$ packets, generation seed = $115$, boundary-referenced confidence $C_C = 2|p - 0.5|$). The canonical rerun is audited separately to reconcile the 37 vs 38 negative-score difference; it is not pooled with or substituted for the frozen baseline.
- **Statistical Testing & Bootstrap Protocol**:
  - The $B = 1000$ bootstrap resamples (seed $7117$, nominal $\alpha = 0.05$) declared in C11.17 were executed to establish baseline distributional confidence intervals.
  - In C11.18, comparative evaluation across policies on the paired 500-packet cohort is performed using paired $t$-tests and Wilcoxon signed-rank tests. These comparisons are exploratory, unadjusted pairwise tests across controlled decision packets, not a prospective clinical cohort.
- **Evaluated Policy Suite**:
  - **B1**: Reliability-Selected Unimodal (Winner-take-all by highest historical reliability prior).
  - **B2**: Uniform Average ($w_i = 1 / |\mathcal{A}|$).
  - **B3**: Confidence Fusion ($\alpha = 1.0, \beta=0, \gamma=0, \eta=0$).
  - **B4**: Confidence + Reliability ($\alpha = 1.0, \beta=1.5, \gamma=0, \eta=0$).
  - **B5**: Confidence + Reliability - Uncertainty ($\alpha = 1.0, \beta=1.5, \gamma=1.0, \eta=0$).
  - **B6**: Full ACARA-U Dynamic Router ($\alpha = 1.0, \beta=1.5, \gamma=1.0, \eta=0.5$).

---

## 3. Pre-Declared Forensic Failure Modes

### Mode 1: Negative DCRI Anomaly
- **Definition**: Cases where the linear epistemic uncertainty penalty exceeds the base fused risk score:
  $$P_U = 0.10 \sum_{i \in \mathcal{A}} U_i > R_{\mathrm{fusion}} \implies \text{DCRI} < 0$$
- **Forensic Questions**:
  1. What is the exact prevalence and distribution of negative DCRI across all policies?
  2. Does negative DCRI produce unintended escalation into non-routine tiers under the thresholding rules?
  3. How does the confidence calibration update shift negative counts between baseline and redesigned rerun cohorts (37 vs 38 packets)?

### Mode 2: Extreme Cross-Modality Conflict Dynamics
- **Definition**: The top 5% extreme disagreement encounters, ranked by maximum pairwise risk disparity:
  $$\Delta_{\max} = \max_{j, k \in \mathcal{A}} |r_j - r_k| \ge Q_{0.95}(\Delta_{\max})$$
- **Forensic Questions**:
  1. How does ACARA-U distribute modality authority weights under extreme pairwise discordance?
  2. Does extreme conflict induce two-tier jumps relative to multi-channel baselines?
  3. How does the analytical mean pairwise distance $\Delta_{\mathrm{mean}} = \frac{1}{3}\sum |r_j - r_k|$ compare to the legacy stored column `css` (which matches $\Delta_{\max}$)?

### Mode 3: Decision Tier Transitions & Reclassification Asymmetries
- **Definition**: Paired tier transitions between Full ACARA-U (B6) and comparator baselines (B1–B5) across all $N=500$ packets:
  $$\Delta_{\mathrm{Tier}}(p, B_k) = \mathrm{Tier}_{B6}(p) - \mathrm{Tier}_{B_k}(p)$$
- **Forensic Questions**:
  1. How frequently does B6 reclassify, upgrade, or downgrade simulated decision packets compared to each baseline?
  2. Does B6 exhibit radical two-tier jumps ($|\Delta_{\mathrm{Tier}}| = 2$), and under which comparators do they occur?
  3. Are reclassification shifts symmetric, or is there a systematic directional bias?

### Mode 4: Uncertainty & Quality Interaction Patterns (Patterns A–D)
- **Pattern A (High Uncertainty Routine Assignment)**: Packets with cumulative uncertainty in the top decile ($U_{\mathrm{sum}} \ge 0.9212$) assigned to Routine Review (Tier 0).
- **Pattern B (Low Quality High Authority)**: Packets where an active modality with degraded quality ($Q_i < 0.60$) receives substantial routing authority ($w_i \ge 0.333$). Evaluated on nominal clean RFC benchmark data ($Q_i \ge 0.85$).
- **Pattern C (Extreme Conflict Invariance)**: Packets in the top 5% conflict bracket that receive identical tier assignments across all policies B1–B6.
- **Pattern D (Decision Boundary Proximity & Near-Threshold Sensitivity)**: Packets residing within a narrow window of $\pm 0.02$ around decision thresholds $\tau_1 = 0.20$ or $\tau_2 = 0.40$.

---

## 4. Upstream Artifact Verification & Provenance
Phase C11.18 ingests the sealed artifacts from Phase C11.17 (`experiments/fusion/decision_evaluation/`):
- `02_evaluation_config.json` (SHA-256: `3ac4db007f8fcbbca61f52e97aa2e2cc65c4d78615eda80cc6d22d8786329511`)
- `03_packet_level_results.csv` (SHA-256: `6c0b755a1b5c0ea526f8fd1dc5df483512eea29118022e5a59824563670a92e5`)
- `05_decision_transitions.csv` (SHA-256: `e3b502d274233ae84dafd982fe6cfccec9fb01824e89a925e4eca9f57328b1a7`)
- `06_statistical_analysis.json` (SHA-256: `b81668a6053f80496076f77ec905ec57f9c387309263d9a3d95f58d5c352f54a`)

And cross-references the canonical rerun artifact:
- `04_canonical_c11_full_metrics.json` (Canonical Redesigned Rerun metrics).

All analyses are executed using paired, non-pseudoreplicated statistical methodology on the full $N = 500$ controlled decision-packet cohort.
