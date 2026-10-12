# Volume 16: ACARA-U Component Ablation Study

## Overview

Volume 16 establishes the definitive component-level attribution and ablation audit of the **ACARA-U v2 Dynamic Multimodal Router** within the FusionMedAI decision fusion research framework.

Following the formal sealing of the baseline configuration ($B6$) in Phase C11.15, Phase C11.16 systematically evaluates the individual contributions of each mathematical component in the routing logit formulation:
$$z_i = \alpha C_i + \beta R_i - \gamma U_i + \eta Q_i$$
alongside the downstream Dynamic Composite Risk Index (DCRI) epistemic uncertainty discount ($\delta^* = 0.10$).

The ablation protocol operates under strict pre-specified frozen parameters on the canonical $N=500$ controlled decision packet cohort (seed 115, SHA-256: `1ff37fbeeba33feae7bf14d57d0df4a10df48ac789edeedf9dbd2fa8121382bb`), evaluating one-at-a-time component deletions without hyperparameter retuning or post-hoc optimization.

> **Volume 16 Status:** 🟢 SEALED & AUDIT-CERTIFIED (12/12 Acceptance Gates Certified under Controlled Benchmark Protocol)

---

## Key Scientific Findings

1. **Exact Baseline Reference Reproduction ($A0 \equiv B6$):**
   The reference candidate $A0$ reproduces the frozen Phase C11.13 / C11.15 benchmark outputs within machine precision ($\text{Mean } R_{\text{fusion}} = 0.289900$, $\text{Mean DCRI} = 0.226506$, $\text{Mean } P_U = 0.063394$, negative DCRI encounters = 37, baseline tier counts: 232 Routine Review, 187 Additional Assessment, 81 Escalation), establishing end-to-end numerical reproducibility.

2. **Component Attribution Hierarchy:**
   Component removal reveals a distinct hierarchy in routing weight redistribution:
   - **Predictive Uncertainty Penalty ($A3, \gamma=0$):** Largest redistribution driver. Removing $\gamma U_i$ surges foot ulcer weight by $+0.116634$ ($95\%$ CI: $[+0.112612, +0.120224]$) and raises mean fused risk by $+0.030842$ ($95\%$ CI: $[+0.027805, +0.033985]$).
   - **Modality Confidence ($A1, \alpha=0$):** Second largest driver. Removing $\alpha C_i$ transfers $+0.098873$ weight into tabular clinical data and reduces mean fused risk by $-0.015139$ ($95\%$ CI: $[-0.018219, -0.011936]$).
   - **Reliability Prior ($A2, \beta=0$):** Moderate contribution ($\Delta w_{\text{clinical}} = +0.028372$, $\Delta R = -0.006193$).
   - **Input Signal Quality Bonus ($A4, \eta=0$):** Minimal contribution under controlled quality ($\Delta w_{\text{retina}} = -0.009019$, $\Delta R = -0.000592$).

3. **Orthogonal Downstream DCRI Discount ($A5, \delta=0$):**
   Removing the downstream uncertainty discount leaves routing weights ($w_i$) and fused risk ($R_{\text{fusion}}$) strictly identical ($\Delta w \equiv 0.000000, \Delta R \equiv 0.000000$), but shifts downstream DCRI upward by exactly $+0.063394$ ($95\%$ CI: $[+0.061150, +0.065409]$), eliminating all 37 negative DCRI occurrences and escalating 36 additional packets into Tier 2 (Escalation for Review).

4. **Regime-Stratified Singleton Invariance:**
   Across all unimodal encounters ($R$, $F$, $C$), active-simplex normalization mathematically forces $w_i \equiv 1.000000$, resulting in zero weight or risk variation ($\Delta w \equiv 0.0, \Delta R \equiv 0.0$) across all router ablations ($A1$–$A4$). Routing terms only actively redistribute weights in multi-modality encounters ($|\mathcal{A}| \ge 2$).

5. **Safety Gate Negative Control Rejection ($A6$):**
   Deliberate fault injection bypassing availability masking was detected in $500/500$ test packets ($100\%$ detection rate) and strictly rejected by the acceptance gate, demonstrating active verifier sensitivity against post-normalization leakage.

---

## Chapter Directory

| Chapter | Title | Primary Focus |
| :--- | :--- | :--- |
| **[Chapter 01](01_Ablation_Protocol.md)** | Ablation Protocol & Methodology | Controlled experimental design, frozen parameter contract, and acceptance gate definitions. |
| **[Chapter 02](02_Variant_Specifications.md)** | Variant Specifications | Mathematical formulations for $A0$–$A6$ and diagnostic baseline $A_{\text{UNIFORM}}$. |
| **[Chapter 03](03_Reference_Reproduction.md)** | Reference B6 Reproduction | Numerical verification of $A0$ against C11.13 / C11.15 frozen benchmarks. |
| **[Chapter 04](04_Routing_Weight_Attribution.md)** | Routing Weight Attribution | Decomposition of simplex weights and component attribution hierarchy. |
| **[Chapter 05](05_Fused_Risk_Propagation.md)** | Fused Risk Propagation | Analysis of how weight reallocations translate into fused risk shifts. |
| **[Chapter 06](06_DCRI_Propagation_and_Discount_Ablation.md)** | DCRI Propagation & Discount Ablation | Orthogonality proof, uncertainty penalty quantification, and negative DCRI impact. |
| **[Chapter 07](07_Decision_Tier_Transitions.md)** | Decision Tier Transitions | 3x3 hypothetical triage transition matrices, stability rates, and escalation dynamics. |
| **[Chapter 08](08_Availability_Regime_Breakdown.md)** | Availability Regime Breakdown | Comprehensive evaluation across 8 clinical availability regimes ($R, F, C, RF, RC, FC, RFC, \text{EMPTY}$). |
| **[Chapter 09](09_Negative_Control_Fault_Injection.md)** | Negative Control Fault Injection | Deliberate availability mask breach ($A6$), verifier detection, and gate rejection. |
| **[Chapter 10](10_Component_Sensitivity_and_Failure_Modes.md)** | Component Sensitivity & Failure Modes | Degenerate edge cases, risk decoupling boundaries, and epistemic uncertainty interactions. |
| **[Chapter 11](11_Statistical_Synthesis.md)** | Statistical Synthesis | Paired bootstrap resampling methodology ($B=1,000$, seed 7116) and confidence intervals. |
| **[Chapter 12](12_Results.md)** | Results & Tabulations | Consolidated reference tables, metric comparisons, and cross-variant summaries. |
| **[Chapter 13](13_Freeze_Report.md)** | Freeze Declaration & Gate Certification | Formal certification audit, SHA-256 artifact manifest, and Phase C11.16 closure. |

---

## Verification & Acceptance Summary

The Phase C11.16 component ablation experiment was verified against 12 formal acceptance gates:
- **C16-01**: Artifact Manifest & Checksum Integrity — **PASSED**
- **C16-02**: Reference B6 Reproduction Invariant — **PASSED**
- **C16-03**: Simplex & Safety Invariant Preservation — **PASSED**
- **C16-04**: Negative Control Fault Rejection ($A6$) — **PASSED**
- **C16-05**: Unimodal Regime Invariance — **PASSED**
- **C16-06**: DCRI Discount Orthogonality ($A5$) — **PASSED**
- **C16-07**: Uncertainty Penalty Directionality ($A3$) — **PASSED**
- **C16-08**: Component Attribution Hierarchy — **PASSED**
- **C16-09**: Decision Tier Transition Integrity — **PASSED**
- **C16-10**: Paired Bootstrap Determinism & Coverage — **PASSED**
- **C16-11**: Fail-Closed EMPTY Regime Invariance — **PASSED**
- **C16-12**: Research Documentation Volume Completeness — **PASSED**

**Overall Audit Score:** 12 / 12 Gates Passed (100.0%)
