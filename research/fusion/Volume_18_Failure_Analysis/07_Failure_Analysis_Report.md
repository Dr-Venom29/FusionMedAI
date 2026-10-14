# Phase C11.18: Comprehensive Failure & Error Analysis Report
**FusionMedAI: Adaptive Context-Aware Reliability Assessment under Uncertainty (ACARA-U)**

---

## 1. Executive Summary & Forensic Scope
Phase C11.18 executes a comprehensive post-hoc failure and error analysis on the frozen Phase C11.17 decision-level evaluation artifacts. In accordance with clinical AI evaluation standards and the frozen C11.18 protocol, this investigation diagnoses, quantifies, and categorizes system edge behaviors under the fully dynamic ACARA-U decision router (Policy B6) and its comparator baselines (B1–B5) across the controlled cohort ($N = 500$, seed $115$).

### Key Findings Matrix
| Diagnostic Dimension | Primary Finding | Clinical / System Impact | Mitigation / Status |
| :--- | :--- | :--- | :--- |
| **Negative DCRI (B6)** | Exactly **37 / 500 (7.40%)** packets exhibit $\text{DCRI} < 0.0$ ($\min = -0.0271$, $\text{mean} = -0.0108$). | Negative DCRI is an expected mathematical property when $P_U > R_{\mathrm{fusion}}$. Mapping to Tier 0 is a direct consequence of $\text{DCRI} < \tau_1 = 0.20$. | Candidate presentation option $\text{DCRI}_{\mathrm{display}} = \max(0, \text{DCRI})$ requires downstream clinical testing; unclamped in audit logs. |
| **Comparator Negative DCRI** | B1 exhibits **223 / 500 (44.6%)** negative DCRI scores; B2: 0, B3: 1, B4: 1, B5: 35. | High negative-score frequency in B1 reflects formula-policy mismatch (single-channel risk minus multi-channel uncertainty). | Multi-channel routing distributes authority across active channels. |
| **Extreme Cross-Modality Conflict** | Top 5% extreme disagreement threshold: $\Delta_{\max} \ge 0.9427$ ($N=25$). | ACARA-U assigns 18 packets to Tier 1 and 7 packets to Tier 2; **zero packets** assigned to Tier 0 under evaluated configuration. | Supported on evaluated 500-packet cohort; establishing that contradictory evidence can never be ignored requires broader validation. |
| **Two-Tier Decision Flip-Flops** | B6 vs B1 exhibits **11 upward 2-tier jumps** (Tier 0 $\to$ Tier 2) and **0 downward jumps** ($N=500$). | Unimodal selection (B1) diverges from multi-channel consensus when secondary modalities detect acute risk. | No two-tier jumps observed between B6 and multi-channel baselines (B2–B5: 0 jumps across $N=500$). |
| **Pattern A: High Uncertainty Routine** | **24 / 500 (4.8%)** packets with $U_{\mathrm{sum}} \ge 0.9212$ fall into Tier 0 (Routine Review). | Empirical diagnosis: Epistemic penalty reduces low-baseline risk ($R_{\mathrm{fusion}} \le 0.08$) below $\tau_1=0.20$. | Proposed downstream candidate mitigation (Policy S Repeat-Exam pathway); not active in frozen B6. |
| **Pattern B: Low Quality Authority** | **0 / 500 (0.0%)** cases meet condition ($Q_i < 0.60, w_i \ge 0.333$). | Nominal RFC inputs maintain clean quality ($Q_i \ge 0.85 > 0.60$); 0 cases trigger under nominal conditions. | Stress-test degradation evaluated separately in Volume 10 ($D_1$–$D_3$). |
| **Pattern C: Extreme Conflict Invariance** | **0 / 25 (0.0%)** extreme-conflict cases have identical tiers across B1–B6. | All extreme conflict cases exhibit policy-dependent tier shifts. | Dynamic routing is decisive in contradictory encounters. |
| **Pattern D: Boundary Proximity & Sensitivity** | **59 / 500 (11.8%)** packets lie within $\pm 0.02$ of decision boundaries ($\tau_1=0.20$: 35, $\tau_2=0.40$: 24). | Near-threshold encounters show boundary proximity / sensitivity; demonstrating true fragility requires perturbation testing. | Flag borderline cases for human-in-the-loop review. |

---

## 2. Forensic Audit of Negative DCRI (Failure Mode 1)

### 2.1 Mathematical Formulation & Occurrence Mechanism
The Diabetic Complication Risk Index (DCRI) is defined as:
$$\text{DCRI}_{0.10} = R_{\mathrm{fusion}} - 0.10 \sum_{i \in \mathcal{A}} U_i$$
Where $R_{\mathrm{fusion}} = \sum_{i \in \mathcal{A}} w_i r_i$ is the simplex-weighted fused risk ($\sum w_i = 1, w_i \ge 0$), and $\sum_{i \in \mathcal{A}} U_i$ is the cumulative epistemic uncertainty across active channels.

A negative DCRI occurs if and only if:
$$P_U = 0.10 \sum_{i \in \mathcal{A}} U_i > R_{\mathrm{fusion}}$$
This condition arises in clinical encounters exhibiting:
1. Low underlying multi-channel risk predictions ($R_{\mathrm{fusion}} < 0.10$).
2. High cumulative model uncertainty ($U_{\mathrm{sum}} > 0.50$, yielding penalty $P_U > 0.05$).

### 2.2 Distributional Metrics across Cohorts
In the sealed C11.17 baseline evaluation cohort ($N=500$, seed $115$):
- **Prevalence**: 37 packets ($7.40\%$).
- **Minimum DCRI**: $-0.027143$.
- **Mean Negative DCRI**: $-0.010750 \pm 0.007612$.
- **Maximum Negative DCRI**: $-0.000095$.
- **Mean $R_{\mathrm{fusion}}$ for Negative Packets**: $0.068849 \pm 0.017842$.
- **Mean $U_{\mathrm{sum}}$ for Negative Packets**: $0.795996 \pm 0.114521$ (yielding mean penalty $0.079600$).

### 2.3 Policy Comparison of Negative Scores
| Policy | Negative DCRI Count | % of Cohort | Min DCRI | Max DCRI | Mean DCRI |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **B1 (Unimodal WTA)** | 223 | 44.60% | -0.0916 | 0.8874 | 0.1916 |
| **B2 (Uniform Avg)** | 0 | 0.00% | 0.0034 | 0.7712 | 0.2334 |
| **B3 (Confidence)** | 1 | 0.20% | -0.0028 | 0.8123 | 0.2508 |
| **B4 (Conf + Rel)** | 1 | 0.20% | -0.0031 | 0.8245 | 0.2551 |
| **B5 (Conf + Rel - Unc)** | 35 | 7.00% | -0.0268 | 0.7981 | 0.2239 |
| **B6 (Full ACARA-U)** | 37 | 7.40% | -0.0271 | 0.8012 | 0.2265 |

**Observation on B1 Negative Frequency**: Unimodal baseline B1 exhibits a high negative-score frequency ($44.60\%$) under the specified multi-channel penalty scope. Because B1 assigns $100\%$ weight to the retina channel ($w_R = 1.0$) while the DCRI formula penalizes cumulative uncertainty across all three available channels ($U_{\mathrm{sum}} = U_R + U_F + U_C$), any encounter where retina predicts low risk ($r_R < 0.10$) while secondary modalities are noisy ($U_F + U_C > 0.80$) produces severe negative deflation. Full ACARA-U (B6) exhibits a much lower negative rate ($7.40\%$) by routing authority dynamically across active channels.

### 2.4 Decision Safety Invariant & Clinical Implications
All 37 negative DCRI packets under Policy B6 map strictly to **Tier 0 (Routine Review)** because $\text{DCRI} < 0.00 < \tau_1 = 0.20$.
- **Mathematical Property**: Negative DCRI is a direct mathematical consequence of subtracting cumulative uncertainty from a low fused risk baseline ($P_U > R_{\mathrm{fusion}}$), rather than an algorithmic anomaly or software malfunction.
- **Threshold Rule Direct Consequence**: Mapping to Tier 0 follows directly from the decision rule ($\text{DCRI} < \tau_1$). While this prevents upward tier inflation, it is a structural outcome of the thresholding architecture and does not inherently prove that no clinically relevant pathology was present.
- **No Negative-to-Escalation Flip**: Negative DCRI never produces an erroneous escalation into Tier 1 or Tier 2.
- **Unclamped Auditability**: Unclamped scores accurately reflect that cumulative epistemic uncertainty exceeds base risk signal, preserving complete transparency in audit logs.

---

## 3. Extreme Cross-Modality Conflict Dynamics (Failure Mode 2)

### 3.1 Stratification & Distribution
Cross-modality conflict is evaluated via maximum pairwise difference:
$$\Delta_{\max} = \max(|r_R - r_F|, |r_R - r_C|, |r_F - r_C|)$$
The 95th percentile cutoff is $\Delta_{\max} = 0.942667$. The top 5% extreme conflict cohort consists of exactly $N = 25$ packets ($\Delta_{\max}$ range: $0.9427$ to $0.9842$).

### 3.2 ACARA-U Triage Behavior under Extreme Conflict
Under extreme discordance, unimodal or poorly regularized fusion policies frequently fail by backing the wrong modality. ACARA-U handles extreme conflict through multi-attribute authority balancing:
- **Tier 2 (Escalation for Human Review)**: 7 packets ($28.0\%$).
- **Tier 1 (Additional Assessment)**: 18 packets ($72.0\%$).
- **Tier 0 (Routine Review)**: **0 packets ($0.0\%$)**.

**Empirical Finding on Evaluated Cohort**: Across all 25 top-5% contradictory encounters in this evaluation cohort, ACARA-U assigned every packet to either Tier 1 or Tier 2, with zero packets assigned to Tier 0 under the evaluated configuration.
- **Scope Boundary**: This empirical result confirms that extreme conflict cases were not dismissed into Routine Review under the tested configuration. However, establishing that contradictory clinical evidence can *never* be ignored under any clinical scenario would require broader external validation across expanded multi-center patient cohorts.

### 3.3 Entropy and Modality Dominance
- Mean routing entropy in the top 5% conflict bracket is $1.042 \pm 0.038$ nats (indicating active multi-channel engagement, well above unimodal zero).
- Mean weights in extreme conflict:
  - $\bar{w}_{\mathrm{retina}} = 0.482 \pm 0.054$
  - $\bar{w}_{\mathrm{foot}} = 0.281 \pm 0.048$
  - $\bar{w}_{\mathrm{clinical}} = 0.237 \pm 0.039$
ACARA-U prevents any single conflicting channel from unilaterally commandeering $100\%$ authority.

---

## 4. Decision Tier Transitions & Reclassification Asymmetries (Failure Mode 3)

### 4.1 Paired Transition Matrix
Comparing Policy B6 against comparator baselines across the $N=500$ cohort:

| Comparison | Identical Tier | Upward Shift ($\Delta_{\mathrm{Tier}} > 0$) | Downward Shift ($\Delta_{\mathrm{Tier}} < 0$) | Total Reclassifications | 2-Tier Up (+2) | 2-Tier Down (-2) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **B6 vs B1 (Unimodal)** | 316 (63.2%) | 78 (15.6%) | 106 (21.2%) | 184 (36.8%) | **11 (2.2%)** | **0 (0.0%)** |
| **B6 vs B2 (Uniform)** | 419 (83.8%) | 43 (8.6%) | 38 (7.6%) | 81 (16.2%) | 0 (0.0%) | 0 (0.0%) |
| **B6 vs B3 (Confidence)** | 443 (88.6%) | 9 (1.8%) | 48 (9.6%) | 57 (11.4%) | 0 (0.0%) | 0 (0.0%) |
| **B6 vs B4 (Conf + Rel)** | 436 (87.2%) | 8 (1.6%) | 56 (11.2%) | 64 (12.8%) | 0 (0.0%) | 0 (0.0%) |
| **B6 vs B5 (w/o Quality)**| 497 (99.4%) | 3 (0.6%) | 0 (0.0%) | 3 (0.6%) | 0 (0.0%) | 0 (0.0%) |

### 4.2 Analysis of Two-Tier Jumps
- **B1 Two-Tier Upward Shifts**: Exactly 11 packets shifted from Tier 0 under B1 to Tier 2 under B6 (with 0 downward two-tier shifts). These encounters exhibit low retinal risk predictions ($r_R \le 0.12$, triggering Tier 0 under unimodal B1) alongside elevated complication risk in secondary modalities ($r_F, r_C \ge 0.75$). ACARA-U integrates this distributed evidence, elevating the simulated packet to Tier 2. This indicates a marked policy difference where unimodal selection diverges from multi-channel aggregation, rather than a clinically confirmed false negative.
- **Zero Two-Tier Jumps against Multi-Channel Baselines**: ACARA-U produces **zero two-tier jumps** against B2, B3, B4, and B5. The transitions between ACARA-U and other multi-channel approaches are orderly, single-step boundary adjustments.

---

## 5. Epistemic Uncertainty & Quality Interaction Patterns (Patterns A–D)

### 5.1 Pattern A: High Uncertainty Routine Assignment
- **Definition**: Cumulative uncertainty in the top decile ($U_{\mathrm{sum}} \ge 0.9212$) assigned to Routine Review (Tier 0).
- **Prevalence**: Exactly **24 / 500 packets (4.80%)**.
- **Empirical Diagnosis**: These encounters feature low baseline fused risk ($R_{\mathrm{fusion}} \in [0.03, 0.09]$). The uncertainty penalty ($0.10 \cdot U_{\mathrm{sum}} \approx 0.095$) offsets the fused risk, yielding negative or near-zero DCRI ($< \tau_1 = 0.20$).
- **Diagnosis vs Proposed Mitigation**: This finding represents an empirical diagnosis of the frozen baseline (Policy B6) rather than a solved state within this phase. Subtracting uncertainty reduces triage priority for cases with low base risk despite high epistemic noise. Safeguard Policy S (Volume 14) is a proposed downstream mitigation (incorporating a Repeat Exam Flag and downgrade suppression), not an integrated component of the frozen B6 configuration evaluated here.

### 5.2 Pattern B: Low Quality High Authority
- **Condition Definition**: Modality quality $Q_i < 0.60$ receiving substantial authority weight $w_i \ge 0.333$.
- **Prevalence**: Exactly **0 / 500 packets (0.00%)**.
- **Nominal Benchmark Context**: Zero packets trigger this condition because nominal clean RFC benchmark data features high-quality inputs across all channels ($Q_R = 0.95, Q_F = 0.90, Q_C = 0.85$). Since all $Q_i \ge 0.85 > 0.60$, the prerequisite low-quality condition is never satisfied under nominal operating conditions. The router's degradation penalty response was separately stress-tested in Volume 10 under synthetic noise injections ($D_2, D_3$).

### 5.3 Pattern C: Extreme Conflict Invariance
- **Definition**: Top 5% extreme conflict packets ($\Delta_{\max} \ge 0.9427$) that retain an identical decision tier across all 6 policies.
- **Prevalence**: Exactly **0 / 25 packets (0.00%)**.
- **Mechanism**: Every single severe conflict packet experiences tier shifts across policies, demonstrating that in contentious multi-modal encounters, router policy selection directly dictates clinical disposition.

### 5.4 Pattern D: Decision Boundary Proximity & Near-Threshold Sensitivity
- **Definition**: Encounters with DCRI scores within $\pm 0.02$ of decision thresholds $\tau_1 = 0.20$ or $\tau_2 = 0.40$.
- **Prevalence**: Exactly **59 / 500 packets (11.80%)**.
  - **Near $\tau_1 = 0.20$**: 35 packets ($7.00\%$) in $[0.18, 0.22]$.
  - **Near $\tau_2 = 0.40$**: 24 packets ($4.80\%$) in $[0.38, 0.42]$.
- **Boundary Proximity vs Fragility**: These 59 packets demonstrate boundary proximity and near-threshold sensitivity under the fixed test inputs. While this proximity indicates susceptibility to small changes in inputs or routing weights, establishing true empirical fragility requires formal perturbation testing across noise models. Borderline encounters should be flagged for human-in-the-loop audit.

---

## 6. Paired Statistical Synthesis (Non-Pseudoreplicated)

To ensure rigorous statistical inference without pseudo-replication, paired non-parametric tests were performed across all 500 matched packets:

| Comparison | Mean $\Delta \text{DCRI}$ | Paired $t$-statistic | $t$-test $p$-value | Wilcoxon $W$-statistic | Wilcoxon $p$-value | Significance ($\alpha=0.05$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **B6 vs B1** | $+0.034863$ | $4.492$ | $8.799 \times 10^{-6}$ | $28531.0$ | $1.264 \times 10^{-5}$ | **Statistically Significant** |
| **B6 vs B2** | $-0.006904$ | $-2.495$ | $1.293 \times 10^{-2}$ | $44012.0$ | $1.200 \times 10^{-2}$ | **Statistically Significant** |
| **B6 vs B3** | $-0.024297$ | $-14.732$ | $2.136 \times 10^{-37}$ | $14022.0$ | $5.286 \times 10^{-33}$ | **Statistically Significant** |
| **B6 vs B4** | $-0.028543$ | $-18.641$ | $3.934 \times 10^{-51}$ | $8921.0$ | $3.106 \times 10^{-43}$ | **Statistically Significant** |
| **B6 vs B5** | $+0.002653$ | $13.518$ | $1.106 \times 10^{-31}$ | $2104.0$ | $7.052 \times 10^{-21}$ | **Statistically Significant** |

All five paired differences are statistically significant at nominal $\alpha = 0.05$. 

### Statistical Scope & Methodological Notes
1. **Unit of Analysis & Generative Scope**: Tests evaluate paired differences across $N = 500$ simulated decision packets generated under fixed seed $115$. The reported $p$-values demonstrate distinct algorithmic score distributions under this controlled benchmark setup; they do not establish improved patient outcomes or clinical efficacy.
2. **Multiple Comparisons**: The five pairwise comparisons against B6 are reported as exploratory, unadjusted tests. If a family-wise error rate correction is applied (e.g., Bonferroni adjusted $\alpha = 0.05 / 5 = 0.01$), all comparisons except B6 vs B2 ($p \approx 0.012$) remain statistically significant.
3. **Bootstrap Protocol Alignment**: The $B = 1000$ bootstrap resamples declared in the C11.17 protocol were used to compute baseline distributional confidence bounds; C11.18 relies on the paired $t$-test and Wilcoxon signed-rank test for paired hypothesis testing.

---

## 7. Candidate Mitigations and Validation Requirements

Phase C11.18 documents observed failure patterns under the frozen evaluation setup. The following candidate mitigations are proposed for downstream testing, but were not integrated into the frozen baseline evaluated here:

### Mitigation 1: Clinician UI Display Rectification with Backend Raw Audit Logging
- **Observed Problem**: 37 packets in baseline B6 exhibit negative DCRI scores down to $-0.0271$, which could cause clinician confusion if displayed as negative risk percentages.
- **Proposed Mitigation**: Render $\text{DCRI}_{\mathrm{display}} = \max(0.0, \text{DCRI})$ in clinician dashboard views, while preserving unclamped raw scores in audit logs.
- **Test Required Before Acceptance**: Clinician usability and comprehension testing to verify that clamping does not obscure severe uncertainty or mislead triage decisions.
- **Status in Current Implementation**: Proposed presentation option; not evaluated in Phase C11.18.

### Mitigation 2: Dual-Pathway Safeguard Gateway (Policy S)
- **Observed Problem**: 24 packets with high cumulative uncertainty ($U_{\mathrm{sum}} \ge 0.9212$) are mapped to Tier 0 Routine Review because linear uncertainty discounting offsets low baseline risk.
- **Proposed Mitigation**: Intercept ambiguous encounters ($U_{\mathrm{sum}} \ge 0.30 \lor \min(Q) < 0.60$), block downward reclassification, hold base $R_{\mathrm{fusion}}$ acuity, and issue a Repeat Exam Flag (`ambiguous_evidence = true`).
- **Test Required Before Acceptance**: Dedicated validation evaluating Policy S against the frozen failure cases and new stress cases, measuring false-downgrade reduction against repeat-exam burden.
- **Status in Current Implementation**: Candidate downstream mitigation analyzed in Volume 14; was not active in the frozen baseline B6 evaluated in C11.18.

### Mitigation 3: Inter-Tier Borderline Encounter Highlighting
- **Observed Problem**: 59 packets lie within $\pm 0.02$ of decision thresholds $\tau_1 = 0.20$ or $\tau_2 = 0.40$, indicating near-threshold sensitivity.
- **Proposed Mitigation**: Render borderline proximity visual flags and confidence intervals in review interfaces to alert reviewers.
- **Test Required Before Acceptance**: Formal perturbation testing across noise models to quantify true empirical fragility and boundary crossing probabilities.
- **Status in Current Implementation**: Proposed review interface feature; perturbation analysis is pending.

> [!IMPORTANT]
> **Scope of Deployment Readiness**: Phase C11.18 is a retrospective forensic error analysis on controlled decision packets. It does not establish clinical safety, prospective multi-center generalization, or readiness for clinical deployment.

---

## 8. Readiness Certification for Phase C11.19
Phase C11.18 documents the specified failure patterns observed in the frozen 500-packet evaluation cohort. The reported automated checks passed the ten declared gates. These findings support proceeding to the documentation and artifact-freeze stage (Phase C11.19 — Final Research Freeze & Deployment Readiness), subject to resolving the limitations and inconsistencies identified in this report. They do not establish clinical safety, generalization to real patients, or readiness for clinical deployment.

- **Reported Software Verification Results**: All 10/10 pre-declared software verification gates passed in automated pipeline testing (`08_verification_report.json`), verifying internal consistency, cryptographic integrity, and tier assignments across the frozen cohort.
- **Independent Clinical Governance Scope**: These results represent reported automated software verification results on pre-declared tests within the frozen simulation benchmark, pending independent external reproduction and multi-site prospective validation.
- **Phase Transition**: The forensic audit of Phase C11.17 is complete, and the research documentation is formally prepared to proceed to **Phase C11.19 — Final Research Freeze & Deployment Readiness**.
