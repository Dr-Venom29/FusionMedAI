# Volume 14 — DCRI Decision Policy Sensitivity & Operating Analysis

> **Phase C11.14 Research Documentation & Policy Evaluation Freeze Report**  
> **Status:** 🟢 SEALED & VERIFIED  
> - **Baseline Decision Cohort**: 12/12 Gates Passed (Baseline Decision Cohort fully sealed and certified)  
> - **Canonical Redesigned Cohort**: 5/5 Invariant Tests Passed (`verification/fusion/dcri_policy_analysis/test_canonical_safeguard.py`, Authoritative Default)  
> - **Policy Test Discovery**: 39/39 Unit, Invariant, and Mutation Tests Passed  
> **Frozen Parameter:** $\delta^* = 0.10$ (`D10`)  
> **Nominal Policy Thresholds:** $\tau_1 = 0.20, \tau_2 = 0.40$  
> **Primary Cohort Profile (Authoritative Default):** Canonical Redesigned Cohort (`experiments/fusion/canonical_redesigned_rerun/01_canonical_redesigned_packet_manifest.json`, SHA-256 `cf877d74e306...`, $N=500$, Seed 115)  
> **Historical Baseline Cohort (Explicit Opt-In):** Baseline Decision Cohort (`experiments/fusion/dcri/packet_manifest.json`, SHA-256 `1ff37fbeeba3...`, $N=500$, Seed 115)  
> **Router State:** Frozen Reference Configuration $\Theta_0 = (\alpha=1.0, \beta=1.5, \gamma=1.0, \eta=0.5)$  
> **Outcome Evaluation Addendum:** $N=5,000$ Synthetic Packets across 10 Seeds (8/8 Outcome Gates Passed)

---

## Executive Summary

Phase C11.14 investigates how the frozen uncertainty-discounted decision index ($\text{DCRI}_{0.10} = R_{\mathrm{fusion}} - 0.10 \sum U_i$) alters hypothetical decision-policy actions relative to the unpenalized fused risk reference policy ($R_{\mathrm{fusion}}$) across predefined operating thresholds and modality availability conditions.

> [!NOTE] **Methodological Scope & Multi-Stage Evaluation**:
> - **Core Operating Analysis (Chapters 01–08)**: Measures tier reclassification rates, workload reduction, and threshold sensitivity on the frozen $N=500$ controlled decision cohort (12/12 gates passed). These establish operational mechanics and workload behavior, not clinical outcome improvement.
> - **Outcome-Grounded Addendum (Chapter 09)**: Scores the decision policies against a known ground-truth oracle target across $N=5,000$ synthetic packets under an illustrative asymmetric clinical cost model, revealing that DCRI's workload reduction entails an increased false downgrade rate among uncertain high-risk cases ($7.61\% \to 12.72\%$).

---

## Key Empirical Findings

1. **Reclassification Rate & Non-Inflationary Invariant**: At standard operating thresholds ($\tau_1=0.20, \tau_2=0.40$), $\text{DCRI}_{0.10}$ reclassifies **$19.2\%$** ($96/500$ packets) into a lower action tier. Zero packets ($0.0\%$) are upgraded, confirming the strict monotonic non-inflationary invariant of additive uncertainty discounting.
2. **Escalation Workload Attenuation**: High-priority escalation review is reduced from **$16.6\%$** ($83/500$ packets) under unpenalized fused risk to **$11.4\%$** ($57/500$ packets) under $\text{DCRI}_{0.10}$—a **$31.33\%$ relative reduction** (26 packets reclassified from Escalation to Additional Assessment; $5.2$ percentage points of the cohort).
3. **Modality-Dependent Reclassification**: Reclassification rates depend on modality composition and aggregate observed uncertainty rather than modality cardinality alone. Single-modality encounters with very low predictive uncertainty (`R` with $U\approx 0.0014$, `C` with $U\approx 0.0386$) exhibit $0.0\%$ reclassification, and dual-modality `RC` exhibits only $1.2\%$ ($6/500$). In contrast, single-modality `F` exhibits $16.8\%$ ($84/500$), and high-uncertainty dual/triple combinations (`RF`, `FC`, `RFC`) exhibit reclassification rates between $19.2\%$ and $32.2\%$.
4. **Perturbation Robustness**: Under $\pm 20\%$ global uncertainty scaling, reclassification rates increase monotonically across the tested grid. Under $\pm 0.02$ threshold boundary jitter, reclassification rates remain tightly bounded, without threshold bifurcation or invariant collapse.

---

## Master Policy Scoreboard (Nominal $\tau_1 = 0.20, \tau_2 = 0.40$)

| Policy Condition | Routine Review (Tier 0: $< 0.20$) | Additional Assessment (Tier 1: $[0.20, 0.40)$) | Escalation (Tier 2: $\ge 0.40$) | Total Reclassified | Monotonic Downgrades | Upgrades |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Policy B: Fused Risk $R_{\mathrm{fusion}}$ (Reference)** | $192$ ($38.4\%$) | $225$ ($45.0\%$) | $83$ ($16.6\%$) | — | — | — |
| **Policy A: $\mathrm{DCRI}_{0.10}$ (Uncertainty-Discounted)** | **$262$ ($52.4\%$)** | **$181$ ($36.2\%$)** | **$57$ ($11.4\%$)** | **$96$ ($19.2\%$)** | **$96$ ($100\%$)** | **$0$ ($0\%$)** |

### Cohort A: Baseline Decision Cohort ($N=500$, Seed 115, SHA-256 `1ff37fbeeba3...`)
| Policy Condition | Routine Review (Tier 0: $< 0.20$) | Additional Assessment (Tier 1: $[0.20, 0.40)$) | Escalation (Tier 2: $\ge 0.40$) | Total Reclassified | Monotonic Downgrades | Upgrades | Blocked Downgrades | Repeat-Exam Flags |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Policy B: Fused Risk ($R_{\mathrm{fusion}}$ Reference)** | $176\ (35.2\%)$ | $207\ (41.4\%)$ | $117\ (23.4\%)$ | — | — | — | — | — |
| **Policy A: $\mathrm{DCRI}_{0.10}$ (Unconstrained)** | **$232\ (46.4\%)$** | **$187\ (37.4\%)$** | **$81\ (16.2\%)$** | **$92\ (18.4\%)$** | **$92\ (100\%)$** | **$0\ (0.0\%)$** | — | — |
| **Policy S: Safeguarded DCRI Gateway** | **$176\ (35.2\%)$** | **$210\ (42.0\%)$** | **$114\ (22.8\%)$** | **$3\ (0.6\%)$** | **$3\ (100\%)$** | **$0\ (0.0\%)$** | **$89 / 92\ (96.7\%)$** | **$432 / 500\ (86.4\%)$** |

### Cohort B: Canonical Redesigned Cohort ($N=500$, Seed 115, SHA-256 `cf877d74e306...`)
| Policy Condition | Routine Review (Tier 0: $< 0.20$) | Additional Assessment (Tier 1: $[0.20, 0.40)$) | Escalation (Tier 2: $\ge 0.40$) | Total Reclassified | Monotonic Downgrades | Upgrades | Blocked Downgrades | Repeat-Exam Flags |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Policy B: Fused Risk ($R_{\mathrm{fusion}}$ Reference)** | $192\ (38.4\%)$ | $225\ (45.0\%)$ | $83\ (16.6\%)$ | — | — | — | — | — |
| **Policy A: $\mathrm{DCRI}_{0.10}$ (Unconstrained)** | **$262\ (52.4\%)$** | **$181\ (36.2\%)$** | **$57\ (11.4\%)$** | **$96\ (19.2\%)$** | **$96\ (100\%)$** | **$0\ (0.0\%)$** | — | — |
| **Policy S: Safeguarded DCRI Gateway** | **$193\ (38.6\%)$** | **$225\ (45.0\%)$** | **$82\ (16.4\%)$** | **$2\ (0.4\%)$** | **$2\ (100\%)$** | **$0\ (0.0\%)$** | **$94 / 96\ (97.9\%)$** | **$432 / 500\ (86.4\%)$** |

---

## Decision-Policy Safeguard Gateway (Policy S)

To address the clinical hazard identified in Chapter 09—where unconstrained DCRI uncertainty discounting causes a $67.1\%$ surge in false downgrades of high-risk cases ($7.61\% \to 12.72\%$)—we evaluate **Policy S (Safeguarded DCRI Gateway)** as a dedicated safety policy layer:

1. **Decoupling Score Calculation from Policy Interpretation**:
   The underlying mathematical definition of DCRI remains strictly unmodified ($\text{DCRI}_{0.10} = R_{\mathrm{fusion}} - 0.10 \sum U_i$). The safeguard operates purely at the decision-tier transition boundary.
2. **Ambiguity Gating Criteria**:
   An encounter is classified as ambiguous if either cumulative predictive uncertainty is elevated or any active sensor exhibits degraded data quality:
   $$\text{Ambiguous}(\mathbf{x}) \iff \left(\sum_{i \in \mathcal{A}} U_i \ge 0.30\right) \lor \left(\min_{i \in \mathcal{A}} Q_i < 0.60\right)$$
3. **Selective Downgrade Suppression & Escalation Preservation**:
   - If evidence is **ambiguous** and DCRI attempts a downward tier transition ($T_{\mathrm{DCRI}} < T_{\mathrm{base}}$), the transition is **blocked**, holding the case at its unpenalized fused risk tier ($T_{\mathrm{final}} = T_{\mathrm{base}}$).
   - Simultaneously, a mandatory diagnostic workflow flag `FLAG_REPEAT_EXAM_AMBIGUOUS_EVIDENCE` is emitted to route the patient for imaging re-acquisition or confirmatory laboratory testing.
   - If evidence is **clear** ($U_{\mathrm{sum}} < 0.30$ and $\min(Q) \ge 0.60$), DCRI downgrades are permitted.
4. **Empirical Results on Frozen Cohorts ($N=500$, Seed 115)**:
   - **Primary Canonical Redesigned Cohort**:
     - **$94$ of $96$ attempted downgrades are blocked ($97.9\%$)**, almost completely eliminating false-downgrade risk from uncertainty penalties.
     - Priority Escalations are preserved at **$82$ cases ($16.4\%$)**, compared to $57$ under unconstrained DCRI and $83$ under base fused risk.
     - Only **$2$ cases ($0.4\%$)** are downgraded under clear evidence.
     - **Repeat-Exam Workload Burden**: **$432 / 500$ encounters ($86.4\%$)** trigger the repeat-exam / ambiguous flag.
   - **Baseline Decision Cohort (Historical Opt-In)**:
     - **$89$ of $92$ attempted downgrades are blocked ($96.7\%$)**.
     - Priority Escalations are preserved at **$114$ cases ($22.8\%$)**, compared to $81$ under unconstrained DCRI and $117$ under base fused risk.
     - Only **$3$ cases ($0.6\%$)** are downgraded under clear evidence.
     - Repeat-exam flag rate is identically **$432 / 500$ ($86.4\%$)**.
   - This makes the operational cost explicit: eliminating false downgrades converts under-triage risk into a substantial diagnostic re-examination workload.

*Artifact Traceability*: Sealed in `experiments/fusion/dcri_policy_analysis/results/safeguarded_policy_results.json`.

---

## Volume Documentation Index

- [Chapter 01 — Scientific Protocol & Research Hypotheses](01_Protocol.md)
- [Chapter 02 — Policy Definitions & Threshold Framework](02_Policy_Definitions.md)
- [Chapter 03 — Decision Threshold & Sensitivity Analysis](03_Threshold_Analysis.md)
- [Chapter 04 — Modality Availability Regime Analysis](04_Regime_Analysis.md) 
- [Chapter 05 — Robustness & Perturbation Analysis](05_Robustness_Analysis.md)
- [Chapter 06 — Statistical Analysis & Confidence Bounds](06_Statistical_Analysis.md)
- [Chapter 07 — Methodological Boundaries & Limitations](07_Limitations.md)
- [Chapter 08 — Freeze Report & Verification Audit](08_Freeze_Report.md)
- [Chapter 09 — Outcome-Grounded Policy Utility & Error Trade-Off Addendum](09_Policy_Outcome_Addendum.md)
