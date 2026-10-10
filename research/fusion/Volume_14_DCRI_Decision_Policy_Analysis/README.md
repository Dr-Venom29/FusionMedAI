# Volume 14 — DCRI Decision Policy Sensitivity & Operating Analysis

> **Phase C11.14 Research Documentation & Policy Evaluation Freeze Report**  
> **Status:** 🟢 SEALED & VERIFIED (12/12 Gates Passed)  
> **Frozen Parameter:** $\delta^* = 0.10$ (`D10`)  
> **Nominal Policy Thresholds:** $\tau_1 = 0.20, \tau_2 = 0.40$  
> **Evaluation Cohort:** Frozen C11 Controlled Multi-Source Cohort ($N=500$, Seed 115)  
> **Router State:** Frozen Reference Configuration $\Theta_0 = (\alpha=1.0, \beta=1.5, \gamma=1.0, \eta=0.5)$

---

## Executive Summary

Phase C11.14 investigates how the frozen uncertainty-discounted decision index ($\text{DCRI}_{0.10} = R_{\mathrm{fusion}} - 0.10 \sum U_i$) alters hypothetical decision-policy actions relative to the unpenalized fused risk reference policy ($R_{\mathrm{fusion}}$) across predefined operating thresholds and modality availability conditions.

Operating on the frozen $N=500$ controlled decision cohort without changing upstream model weights or router coefficients, Phase C11.14 evaluates a 3-tiered hypothetical action taxonomy (**Routine Review**, **Additional Assessment**, and **Escalation for Review**) across a pre-specified 25-pair threshold sensitivity grid, all 7 active modality regimes, and controlled uncertainty/threshold perturbations.

---

## Key Empirical Findings

1. **Reclassification Rate & Non-Inflationary Invariant**: At standard operating thresholds ($\tau_1=0.20, \tau_2=0.40$), $\text{DCRI}_{0.10}$ reclassifies **$18.4\%$** ($92/500$ packets, $95\%$ Wilson CI: $[15.25\%, 22.03\%]$) into a lower action tier. Zero packets ($0.0\%$) are upgraded, confirming the strict monotonic non-inflationary invariant of additive uncertainty discounting.
2. **Escalation Workload Attenuation**: High-priority escalation review is reduced from **$23.4\%$** ($117/500$ packets) under unpenalized fused risk to **$16.2\%$** ($81/500$ packets) under $\text{DCRI}_{0.10}$—a **$30.77\%$ relative reduction** (36 packets reclassified from Escalation to Additional Assessment; 7.2 percentage points of the cohort).
3. **Modality-Dependent Reclassification**: Reclassification rates depend on modality composition and aggregate observed uncertainty rather than modality cardinality alone. Single-modality encounters with very low predictive uncertainty (`R` with $U\approx 0.0014$, `C` with $U\approx 0.0386$) exhibit $0.0\%$ reclassification, and dual-modality `RC` exhibits only $1.2\%$ ($6/500$). In contrast, single-modality `F` exhibits $16.8\%$ ($84/500$), and high-uncertainty dual/triple combinations (`RF`, `FC`, `RFC`) exhibit reclassification rates between $18.4\%$ and $32.2\%$.
4. **Perturbation Robustness**: Under $\pm 20\%$ global uncertainty scaling, reclassification rates increase monotonically across the tested grid ($15.4\%\text{--}21.4\%$). Under $\pm 0.02$ threshold boundary jitter, reclassification rates remain tightly bounded ($18.2\%\text{--}18.6\%$), without threshold bifurcation or invariant collapse.

---

## Master Policy Scoreboard (Nominal $\tau_1 = 0.20, \tau_2 = 0.40$)

| Policy Condition | Routine Review (Tier 0: $< 0.20$) | Additional Assessment (Tier 1: $[0.20, 0.40)$) | Escalation (Tier 2: $\ge 0.40$) | Total Reclassified | Monotonic Downgrades | Upgrades |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Policy B: Fused Risk $R_{\mathrm{fusion}}$ (Reference)** | $176$ ($35.2\%$) | $207$ ($41.4\%$) | $117$ ($23.4\%$) | — | — | — |
| **Policy A: $\mathrm{DCRI}_{0.10}$ (Uncertainty-Discounted)** | **$232$ ($46.4\%$)** | **$187$ ($37.4\%$)** | **$81$ ($16.2\%$)** | **$92$ ($18.4\%$)** | **$92$ ($100\%$)** | **$0$ ($0\%$)** |

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
