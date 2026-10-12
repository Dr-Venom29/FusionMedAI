# Chapter 12: Comprehensive Results Summary

## 1. Executive Summary
This report summarizes the empirical outcomes of **Phase C11.17: Final Decision-Level Evaluation of ACARA-U** across the canonical $N=500$ evaluation cohort (seed 115). All quantitative values are derived directly from the canonical artifacts (`03_canonical_records.json`, `04_summary_metrics.json`, `06_statistical_analysis.json`) and verified by the independent verification suite (`08_verification_report.json`).

### Primary Decision Metrics Across All Evaluated Policies ($N=500$)
| Metric / Policy | B1 (Unimodal WTA) | B2 (Uniform Avg) | B3 (Confidence) | B4 (Conf + Rel) | B5 (Conf+Rel-Unc) | B6 (Full ACARA-U) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mean Fused Risk $\overline{R}_{\text{fusion}}$** | 0.255037 | 0.296804 | 0.314197 | 0.318442 | 0.287247 | **0.289900** |
| **Mean DCRI $\overline{\text{DCRI}}$** | 0.191643 | 0.233410 | 0.250803 | 0.255049 | 0.223853 | **0.226506** |
| **Mean Penalty $\overline{P}_U$** | 0.063394 | 0.063394 | 0.063394 | 0.063394 | 0.063394 | **0.063394** |
| **Routing Entropy $\overline{H}$** | 0.000000 | 1.098612 | 1.061442 | 1.052013 | 1.022036 | **1.011041** |
| **Routine Review (Tier 0)** | 275 (55.0%) | 208 (41.6%) | 202 (40.4%) | 198 (39.6%) | 234 (46.8%) | **232 (46.4%)** |
| **Additional Assessment (Tier 1)** | 84 (16.8%) | 240 (48.0%) | 208 (41.6%) | 207 (41.4%) | 186 (37.2%) | **187 (37.4%)** |
| **Escalation (Tier 2)** | 141 (28.2%) | 52 (10.4%) | 90 (18.0%) | 95 (19.0%) | 80 (16.0%) | **81 (16.2%)** |
| **Negative DCRI Count** | 223 (44.6%) | 0 (0.0%) | 1 (0.2%) | 1 (0.2%) | 35 (7.0%) | **37 (7.4%)** |

## 2. Key Empirical Findings Across Dimensions

1. **Reference Invariant Concordance**: Full ACARA-U (B6) reproduces reference invariants within $10^{-6}$ tolerance: Mean $R_{\text{fusion}} = 0.289900$, Mean $\text{DCRI} = 0.226506$, and exactly 37 negative DCRI records ($7.4\%$).
2. **Quality Bonus Routing Shift (B6 vs B5)**: Adding the adaptive quality bonus weight ($\eta = 0.5$) shifts mean fused risk by $\Delta = +0.002653$ (95% CI: $[+0.002255, +0.003063]$) across identical inputs, reclassifying 3 borderline encounters and redirecting weight toward higher-quality channels ($\Delta w_R = +0.015797$). This isolates an observable routing shift; it does not demonstrate improved clinical diagnostic accuracy.
3. **Availability Mask Enforcement**: Across 16,000 availability regime records (4 policies $\times$ 8 regimes $\times$ 500 packets), zero inactive weight leaks were observed ($w_m \le 10^{-7}$). Unimodal singletons collapse mathematically to unity routing across all 6,000 tested singleton records, and empty inputs strictly trigger the fail-closed software fallback across 2,000 checked empty records.
4. **Input Degradation Under-Escalation**: As input corruption increases from D0 to D3, mean uncertainty penalty climbs from $0.063394$ to $0.101519$, while mean DCRI falls from $0.226506$ to $0.175799$. Consequently, Tier 2 escalations decrease from 16.2% to 11.0% while Tier 0 routine reviews increase from 46.4% to 56.0%. This highlights a structural property of subtractive uncertainty formulations requiring safety investigation in Phase C11.18.
5. **Conflict Stratification**: In tri-modal encounters ($N=500$), continuous discordance stratifies into LOW ($N=45$, $0.0\%$ Tier 2), MODERATE ($N=92$, $1.1\%$ Tier 2), and HIGH ($N=363$, $22.0\%$ Tier 2) with strictly monotonic escalation rates.
6. **Calibration Condition**: Probability calibration produces a modest upward risk and DCRI shift ($\Delta = +0.004615$, 95% CI: $[+0.003544, +0.005752]$), leaving 96.8% (484/500) of patient decision tiers invariant while reclassifying 16 encounters (12 upgrades, 4 downgrades).

## 3. Scope & Research Limitations
1. **Evaluation Cohort**: Results are evaluated across a fixed synthetic cohort of $N=500$ packets under deterministic seed 115. They do not constitute an evaluation on real-world clinical patient populations.
2. **Absence of Clinical Gold Standards**: The evaluation dataset does not include verified patient outcomes or pathologist-confirmed diagnostic ground truth. Observed tier assignments reflect algorithmic policy mappings under frozen thresholds ($\tau_1=0.20, \tau_2=0.40$), not clinically validated triage accuracy.
3. **Subtractive Uncertainty Limitation**: Decreased escalation under degraded inputs demonstrates that subtracting epistemic uncertainty lowers assigned decision tiers. This trade-off must be addressed before any clinical deployment claims.
