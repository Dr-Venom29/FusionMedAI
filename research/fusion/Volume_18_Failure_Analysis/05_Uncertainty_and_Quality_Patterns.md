# Chapter 05: Epistemic Uncertainty & Input Quality Interaction Patterns

## 1. Pattern A: High Uncertainty Routine Assignment
- **Criteria**: Cumulative uncertainty in the top decile ($U_{\mathrm{sum}} \ge 0.9212$) assigned to Routine Review (Tier 0).
- **Prevalence**: Exactly **24 / 500 packets (4.80%)**.
- **Mechanism**: Low baseline disease risk ($R_{\mathrm{fusion}} \in [0.03, 0.09]$) combined with high uncertainty penalty ($0.10 \cdot U_{\mathrm{sum}} \approx 0.095$) forces DCRI below zero or near zero, mapping to Tier 0.
- **Status & Scope**: This is an observed failure pattern in the frozen B6 baseline. Safeguard Policy S is a proposed downstream mitigation described in Volume 14; Phase C11.18 does not establish that it was applied to, or resolved the pattern in, the evaluated configuration. Its behavior must be verified separately before any effectiveness claim is made.

## 2. Pattern B: Low Quality High Authority
- **Criteria**: Modality quality $Q_i < 0.60$ receiving routing weight $w_i \ge 0.333$.
- **Prevalence**: Exactly **0 / 500 packets (0.00%)** in the clean RFC evaluation cohort.
- **Scope & Interpretation**: Zero cases trigger this condition because the nominal clean RFC benchmark data supplies high-quality inputs across all channels ($Q_R = 0.95, Q_F = 0.90, Q_C = 0.85 > 0.60$). Therefore, zero qualifying cases in this cohort is the expected consequence of the input quality values; it does not constitute empirical evidence that the router handles low-quality inputs correctly under real-world noise.
- **Separate Degradation Evidence**: Router attenuation under severe input noise was evaluated separately in Volume 10 under synthetic degradation models ($D_1$–$D_3$); those stress-test results are separate from this nominal clean-cohort evaluation.

## 3. Pattern C: Extreme Conflict Invariance
- **Criteria**: Top 5% extreme conflict encounters ($\\Delta_{\max} \ge 0.9427$) where all 6 policies produce identical tier assignments.
- **Prevalence**: Exactly **0 / 25 packets (0.00%)**.
- **Mechanism**: 100% of extreme conflict cases experience policy-dependent tier disagreements across B1-B6, confirming that router design directly dictates clinical disposition in contradictory cases.

## 4. Pattern D: Decision Boundary Proximity & Near-Threshold Sensitivity
- **Criteria**: Encounters with DCRI scores within $\pm 0.02$ of decision thresholds $\tau_1 = 0.20$ or $\tau_2 = 0.40$.
- **Prevalence**: Exactly **59 / 500 packets (11.80%)**.
  - **Near $\tau_1 = 0.20$**: 35 packets (7.00%) in $[0.18, 0.22]$.
  - **Near $\tau_2 = 0.40$**: 24 packets (4.80%) in $[0.38, 0.42]$.
- **Boundary Proximity vs Fragility**: Proximity alone indicates near-threshold sensitivity under the fixed test inputs. Establishing empirical fragility requires controlled perturbation testing across noise distributions and measuring empirical flip frequencies, which was not performed in this phase.
- **Workflow Consideration**: Flagging borderline encounters in review interfaces could alert clinicians to proximity to decision boundaries.
