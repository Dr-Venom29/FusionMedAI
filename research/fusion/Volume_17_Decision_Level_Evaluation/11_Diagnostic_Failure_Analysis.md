# Chapter 11: Diagnostic Failure Analysis

## 1. Scope of Diagnostic Failure Analysis
While all pre-declared verification gates passed within the frozen evaluation environment, evaluating an algorithmic pipeline requires explicit enumeration and characterization of operational failure modes, boundary edge conditions, and computational anomalies observed across the test cohorts.

## 2. Identified Operational Anomalies and Policy Discrepancies

### 2.1 Anomaly 1: Negative DCRI Scores Under Low Disease Risk
- **Mathematical Description**: When the unweighted uncertainty penalty exceeds the predicted fused risk ($P_U > R_{\text{fusion}}$), the subtractive DCRI formula yields a negative value ($\text{DCRI} < 0$).
- **Observed Prevalence in B6**: Exactly 37 / 500 packets (7.4%) in the reference cohort (seed 115).
- **Observed Numerical Range**: Minimum $\text{DCRI} = -0.027143$, mean of negative subset $= -0.010750$.
- **Trigger Condition**: Occurs whenever $R_{\text{fusion}} < 0.10 \cdot \sum_{i \in \mathcal{A}} U_i$. In practice on this cohort, this occurs when baseline fused risk is modest ($R_{\text{fusion}} < 0.05$) while active modalities output substantial epistemic uncertainties ($\sum U_i > 0.50$).
- **Threshold Mapping Behavior**: Under the frozen decision thresholds ($\tau_1 = 0.20, \tau_2 = 0.40$), all 37 negative scores map to Tier 0 (Routine Review) because $-0.027143 < 0.20$.
- **Safety & Interpretation Boundary**: Although these records map to Tier 0 under the current threshold definitions, subtracting uncertainty from disease risk to produce a negative score requires careful interpretation: reducing a risk score due to high uncertainty is counter-intuitive in safety-critical screening. Prospective proposals (such as zero-floor clamping $\max(0, \text{DCRI})$) alter the score distribution and must be rigorously evaluated for audit trail consistency, downstream consumer expectations, and boundary effects before deployment.

### 2.2 Anomaly 2: Extreme Unimodal Volatility and Two-Tier Discrepancy (B1 vs B6)
- **Description**: The winner-take-all comparator B1 selects a single channel (Retina in tri-modal encounters), discarding complementary sensor inputs.
- **Observed Discrepancy**: B1 assigns negative DCRI to 223 / 500 packets (44.6%) and diverges from B6 on 184 / 500 packets (36.8%).
- **Two-Tier Transitions**: Exactly 11 packets jump across two tiers between B6 and B1. Specifically, all 11 cases are assigned to Tier 2 (Escalation) by B6, but are assigned to Tier 0 (Routine Review) by B1 (with 0 cases moving from B6 Tier 0 to B1 Tier 2).
- **Pairwise Comparison Observation**: Among multi-channel comparators (B2, B3, B4, B5), zero two-tier jumps ($0 \leftrightarrow 2$) were observed relative to B6. Two-tier discordance is isolated entirely to the unimodal winner-take-all comparator (B1), where single-channel reliance can suppress escalations triggered by multi-channel consensus.

### 2.3 Operational Handling: Complete Modality Omission
- **Condition**: Encounter presented with zero active channels ($\mathcal{A} = \emptyset$, Regime EMPTY).
- **Software Fallback**: The evaluation engine enforces zero weights ($w_i = 0.0$), zero fused risk ($R_{\text{fusion}} = 0.0$), zero penalty ($P_U = 0.0$), zero DCRI ($\text{DCRI} = 0.0$), Tier 0, and returns `status = NO_MODALITY_AVAILABLE`.
- **Deployment Boundary**: This handling represents a defined software fallback within the evaluation code. For operational clinical deployment, downstream clinical workflow systems must intercept `NO_MODALITY_AVAILABLE` packets, halt automated pipeline processing, and route the patient encounter to manual intake to avoid unreviewed discharge.

### 2.4 Anomaly 4: Degradation-Induced Under-Escalation (Primary Safety Concern)
- **Observed Phenomenon**: As input degradation severity increases from D0 to D3, mean uncertainty penalty increases from $0.063394$ to $0.101519$, driving mean DCRI downward from $0.226506$ to $0.175799$.
- **Action Tier Consequence**: Tier 2 escalations decrease from 81 (16.2%) at D0 to 55 (11.0%) at D3, while Tier 0 routine reviews increase from 232 (46.4%) to 280 (56.0%).
- **Clinical Implication**: In clinical practice, high input corruption and uncertainty should prompt increased caution (more clinical review), whereas subtractive DCRI reduces the score and moves patients into lower review tiers. This under-escalation dynamic is a major structural concern of subtractive uncertainty formulations and represents the highest-priority topic for subsequent evaluation phases.

## 3. Actionable Research Priorities for Phase C11.18
The following targeted analyses are prioritized for Phase C11.18:
1. **Priority 1 (Degradation Escalation Dynamic)**: Investigate alternative mathematical formulations (such as additive uncertainty premiums for tier assignment vs subtractive discounts for confidence intervals) to ensure input corruption triggers human review rather than routine discharge.
2. **Priority 2 (Negative DCRI Clinical Alignment)**: Conduct clinical panel reviews of the 37 negative DCRI cases to determine whether low-risk high-uncertainty encounters should be routed to Tier 1 (Additional Assessment) instead of defaulting to Tier 0.
3. **Priority 3 (Conflict Boundary Sensitivity)**: Evaluate decision tier transitions across borderline conflict cases ($\Delta_{\max} \approx 0.20$ and $\Delta_{\max} \approx 0.35$) under localized perturbations.
