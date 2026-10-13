# Outcome-Grounded Policy Utility and Error Trade-Offs: Synthetic Evaluation Addendum

> **Evaluation Type:** Independent Synthetic Oracle Ground-Truth Utility Evaluation  
> **Status:** 🟢 VERIFIED (8/8 Outcome-Level Invariant Gates Passed on Dedicated $N=5,000$ Cohort)  
> **Evaluation Sample:** $N=5,000$ Confirmatory Packets (10 Independent Seeds `301`–`310`)  
> **Decision Parameters:** $\delta^* = 0.10$, $\tau_1 = 0.20, \tau_2 = 0.40$  

---

## 1. Context & Scientific Objective

Decision policy analyses on the $N=500$ controlled cohorts established that applying the uncertainty-discounted decision index ($\mathrm{DCRI}_{0.10} = R_{\mathrm{fusion}} - 0.10 \sum U_i$) reduced high-urgency escalation assignments by approximately $31\%$.


However, in clinical decision-making, reducing specialist referrals is only beneficial if it does not systematically miss genuinely high-risk patients. This addendum evaluates:
> **Does DCRI uncertainty discounting improve decision utility when scored against known oracle risk targets under an explicit clinical action-cost matrix, and what is the trade-off in false downgrades of high-risk cases?**

---

## 2. Illustrative Synthetic Cost Matrix & Utility Model

To formalize the trade-offs of policy assignment errors in simulation, we evaluate actions against the ground-truth oracle tier under a prespecified illustrative synthetic cost model:

| Assigned Action Tier | True Low Risk ($Y^* < 0.20$) | True Moderate Risk ($0.20 \le Y^* < 0.40$) | True High Risk ($Y^* \ge 0.40$) |
| :--- | :---: | :---: | :---: |
| **Tier 0 (Routine Review)** | $0.0$ (Correct) | $1.0$ (Under-assessment) | **$5.0$ (Critical Missed Escalation)** |
| **Tier 1 (Additional Assessment)** | $0.5$ (Unnecessary Review) | $0.0$ (Correct) | **$2.5$ (Delayed Escalation)** |
| **Tier 2 (Escalation for Specialist)** | $2.0$ (Unnecessary Escalation) | $1.0$ (Excess Escalation) | $0.0$ (Correct) |

> [!NOTE]
> This cost matrix represents a prespecified illustrative synthetic evaluation model reflecting standard clinical triage asymmetries; it has not been calibrated or validated as an empirical clinical utility function.

---

## 3. Confirmatory Evaluation Findings ($N=5,000$ Synthetic Packets, $3,232$ Oracle High-Risk)

| Metric / Dimension | Fused Risk Policy ($R_{\mathrm{fusion}}$) | DCRI Policy ($\mathrm{DCRI}_{0.10}$) | Net Policy Impact |
| :--- | :---: | :---: | :---: |
| **Total High-Risk Synthetic Packets ($Y^* \ge 0.40$)** | $3,232$ | $3,232$ | Ground-truth baseline |
| **Correct High-Risk Escalations (Tier 2)** | **$2,986$ ($92.39\%$)** | $2,821$ ($87.28\%$) | $-165$ packets ($-5.11\%$) |
| **False Downgrades of High Risk (Tier 0 or 1)** | **$246$ ($7.61\%$)** | $411$ ($12.72\%$) | **$+165$ false downgrades ($+67.07\%$ relative surge)** |
| **Expected Decision Cost Loss ($\mathcal{L}_{\mathrm{cost}}$)** | **$0.2100$** | $0.3040$ | **$+0.0940$ ($+44.76\%$ higher decision loss)** |

### Uncertainty-Stratified False Downgrade Breakdown

| Uncertainty Stratum | Total High-Risk | Fused Risk False Downgrades | DCRI False Downgrades | Surge in False Downgrades |
| :--- | :---: | :---: | :---: | :---: |
| **Low Uncertainty ($U < 0.10$)** | $599$ | $23$ ($3.84\%$) | $35$ ($5.84\%$) | $+12$ ($+52.2\%$) |
| **Moderate Uncertainty ($0.10 \le U < 0.30$)** | $2,132$ | $170$ ($7.97\%$) | $272$ ($12.76\%$) | $+102$ ($+60.0\%$) |
| **High Uncertainty ($U \ge 0.30$)** | $501$ | $53$ ($10.58\%$) | $104$ ($20.76\%$) | **$+51$ ($+96.2\%$ — near doubling)** |

---

## 4. Fundamental Simulation & Policy Trade-Off

1. **Uncertainty Discounting Asymmetry:**
   Additive uncertainty discounting ($\mathrm{DCRI} = R_{\mathrm{fusion}} - \delta \sum U_i$) is mathematically non-inflationary: it can only shift packets downward. While this filters out false-positive escalations when low-risk instances have elevated noise, it simultaneously pushes uncertain but genuinely high-risk synthetic cases below the escalation threshold ($\tau_2 = 0.40$).

2. **Severe Surge in High-Uncertainty Regimes:**
   In encounters characterized by high predictive uncertainty ($U \ge 0.30$), the false downgrade rate under DCRI surges from $10.58\%$ ($53/501$) to **$20.76\%$ ($104/501$)**—nearly doubling the number of under-triaged high-risk synthetic packets.

3. **Decision Loss Under Asymmetric Costs:**
   Because failing to escalate a high-risk instance carries a substantially higher penalty ($2.5\times\text{--}5.0\times$) than unnecessary low-risk review ($0.5\times\text{--}2.0\times$), the increased false downgrade rate ($7.61\% \to 12.72\%$) increases total simulated decision loss from $0.2100$ to $0.3040$ ($+44.8\%$).

4. **Policy Guidance & Simulation Scope:**
   Uncertainty discounting should not be interpreted as an unconditional mechanism for improving clinical decision-making. In high-stakes medical triage where false negatives are costlier than false positives, raw fused risk $R_{\mathrm{fusion}}$ provides superior safety against high-risk under-triage in simulation, while unconstrained DCRI should not be deployed without safeguards. These findings represent simulated decision-policy dynamics and do not establish clinical safety or triage efficacy for human populations.

---

## 5. Decision-Policy Safeguard Resolution (Policy S)

To resolve the false-downgrade surge observed in this synthetic outcome evaluation without abandoning uncertainty-aware decision support or arbitrarily retuning $\delta$, the framework introduces **Policy S (Safeguarded DCRI Gateway)**:

1. **Root-Cause Targeting**:
   The breakdown in Section 3 demonstrates that false downgrades are heavily concentrated in ambiguous regimes ($U \ge 0.30$ surge: $+96.2\%$). In these encounters, epistemic uncertainty suppresses the numerical score below $\tau_2 = 0.40$ even when the underlying clinical signal is genuinely elevated.
2. **Safeguard Rule**:
   - Gating condition: $\text{Ambiguous}(\mathbf{x}) \iff (\sum U_i \ge 0.30) \lor (\min Q_i < 0.60)$ (pre-declared experimental operating thresholds).
   - Reclassification blocking: If ambiguous and $T_{\mathrm{DCRI}} < T_{\mathrm{base}}$, set $T_{\mathrm{final}} = T_{\mathrm{base}}$ and emit `REPEAT_EXAM_AMBIGUOUS_EVIDENCE`.
3. **Cross-Reference: Policy S Outcomes Across Dedicated $N=500$ Cohorts**:
   The dedicated $N=500$ decision-policy cohorts evaluate the exact mechanical behavior of this gateway:
   - **Baseline Decision Cohort ($N=500$, Seed 115, SHA-256 `1ff37fbeeba3...`)**:
     Policy S blocks **$89 / 92$ attempted downgrades ($96.7\%$)**, preserving $114 / 117$ high-priority specialist escalations ($97.4\%$).
   - **Canonical Redesigned Cohort ($N=500$, Seed 115, SHA-256 `cf877d74e306...`)**:
     Policy S blocks **$94 / 96$ attempted downgrades ($97.9\%$)**, preserving $82 / 83$ high-priority specialist escalations ($98.8\%$).
   - **Operational Re-Testing Workload**:
     In both cohorts, exactly **$432 / 500$ encounters ($86.4\%$)** emit the repeat-exam flag.
   
   > [!NOTE]
   > These metrics quantify the operational behavior of the safeguard rule on controlled decision packets. They demonstrate that under-triage suppression converts false downgrades into diagnostic re-examination demand; they do not constitute proof of clinical safety or diagnostic efficacy in prospective human patient care.

