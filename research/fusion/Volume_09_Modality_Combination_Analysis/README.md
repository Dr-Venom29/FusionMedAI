# Volume 09: Modality-Combination Distribution & Tail-Robustness Analysis
 
> **Status**: **SEALED & FROZEN**  
> **Executive Summary**: Comprehensive investigation into how modality combination frequencies impact ACARA-U behavior, routing entropy, and decision-level risk stability across balanced (`D1_BALANCED`), moderate head-tail (`D2_MODERATE_HEAD_TAIL`), and heavy long-tail (`D3_STRONG_LONG_TAIL`) regimes.

---

## 1. Volume Overview & Research Highlights

- **Pre-Specified Distribution Configurations**: 3 controlled frequency distributions across $N=500$ decision packets ($\text{seed}=115$ for exploratory baseline; $S=30$ cohorts, $\text{seeds } 401\text{--}430$ for confirmatory evaluation):
  - **D1 (Balanced)**: $P(c) = 1/7 \approx 14.29\%$ per combination ($HTR = \text{N/A}$, no tail).
  - **D2 (Moderate Tail)**: RFC 35%, RF 25%, RC 15%, FC 10%, R 6%, F 5%, C 4% ($HTR = 4.00$).
  - **D3 (Strong Tail)**: RFC 50%, RF 25%, RC 10%, FC 8%, R 4%, F 2%, C 1% ($HTR = 10.71$).
- **Simplex & Safety Verification**: $0$ routing invariant violations over 1,500 single-cohort trials and 15,000 multi-cohort trials; active weights strictly normalize to $1.000000$ and inactive channels receive $0.000000$.
- **Observed Point-Estimate Tail Sensitivity (Exploratory Seed 115)**: ACARA-U achieved the lowest point-estimate tail risk deviation ($D_{\text{tail}} = 0.1883$ in D2, $0.1876$ in D3) among evaluated soft-weighting baselines (B2 $0.1925/0.1945$, B3 $0.1959/0.1974$, B4 $0.1967/0.1973$, B5 $0.1889/0.1889$), with a small margin over B5 ($95\%$ CIs crossing zero).
- **Multi-Cohort Confirmatory Evaluation ($S=30$ Cohorts, $N=15,000$ Packets)**: Evaluating 30 independent cohorts evaluated tail sensitivity across both micro and macro estimands:
  - *Primary Endpoint (D3 Micro, Packet-Weighted)*: $\Delta D_{\text{tail}} = +0.000181$, $95\%$ Hierarchical Cluster CI: $[-0.000024, +0.000377]$ (crossing zero; $t=2.4194, p=0.022$).
  - *Sensitivity Endpoint (D3 Macro, Combination-Weighted)*: $\Delta D_{\text{tail}}^{\text{macro}} = +0.000123$, $95\%$ Hierarchical CI: $[-0.000093, +0.000344]$ (crossing zero; $t=1.4645, p=0.154$).
  - *Secondary Endpoint (D2 Micro)*: $\Delta D_{\text{tail}} = +0.000143$, $95\%$ CI: $[-0.000025, +0.000300]$ (crossing zero; $t=2.2105, p=0.035$).
  - *Conclusion*: Under the prespecified hierarchical-bootstrap decision rule, the primary endpoint is `INCONCLUSIVE_NOT_STATISTICALLY_DISTINGUISHABLE`. Cohort-level bootstrap and parametric sensitivity analyses indicate a small positive difference favoring B5 numerically, but the primary hierarchical confidence interval includes zero. These results do not establish practical superiority of B6 or satisfy the hierarchical-interval criterion for inferiority. Equivalence is not established.
- **Fail-Closed Verification**: Zero-modality inputs ($\text{EMPTY}$) unconditionally return `NO_MODALITY_AVAILABLE` with $R_{\text{fusion}} = 0.0$ and $\text{DCRI} = 0.0$.
- **Independent Verification Suite**: Strengthened verifier passes 8/8 gates (`CA-01` through `CA-08`) with direct weight inspection and ground-up reconstruction.

---

## 2. Chapter Index & Navigation

1. [**01_Combination_Protocol.md**](01_Combination_Protocol.md): Pre-specified research questions (RQ1–RQ7), hypotheses (H1–H3), and methodological boundaries.
2. [**02_Distribution_Definition.md**](02_Distribution_Definition.md): Mathematical definitions of D1, D2, D3, partition properties, sample sizes, and deterministic stratified assignment.
3. [**03_Head_Tail_Definition.md**](03_Head_Tail_Definition.md): Canonical rank-based tiering (HEAD=Rank 1–2, MIDDLE=Rank 3–4, TAIL=Rank 5–7), Head-to-Tail Ratio ($HTR$).
4. [**04_Combination_Level_Metrics.md**](04_Combination_Level_Metrics.md): Combination-level properties, weight distribution, routing entropy, simplex adherence.
5. [**05_Uncertainty_Analysis.md**](05_Uncertainty_Analysis.md): Uncertainty scaling across cardinality, $U_{\text{sum}}$ behavior, DCRI interaction.
6. [**06_Conflict_and_Risk_Analysis.md**](06_Conflict_and_Risk_Analysis.md): Conflict metrics ($\Delta_{\max}, \Delta_{\text{mean}}, \sigma_w$), risk distribution shifts across combinations.
7. [**07_Baseline_Tail_Comparison.md**](07_Baseline_Tail_Comparison.md): Comparative evaluation against Baselines B1–B6, tail sensitivity $D_{\text{tail}}$, trade-off between head and tail performance.
8. [**08_Tail_Robustness.md**](08_Tail_Robustness.md): Tier contrast metrics ($\Delta \sigma(R), \Delta \sigma(\text{DCRI}), \overline{\Delta R_{\text{tail}}}, \overline{\Delta \text{DCRI}_{\text{tail}}}$), sensitivity analysis.
9. [**09_Statistical_Analysis.md**](09_Statistical_Analysis.md): 1,000-resample bootstrap confidence intervals ($95\%$ CI), hypothesis evaluation outcomes.
10. [**10_Results.md**](10_Results.md): Comprehensive summary tables, key empirical findings, cross-distribution stability.
11. [**11_Freeze_Report.md**](11_Freeze_Report.md): 8/8 verification gates, cryptographic SHA-256 manifest certification, freeze sign-off.
