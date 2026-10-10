# Chapter 05 — Robustness & Perturbation Analysis

## 1. Uncertainty Scaling Perturbations ($U_{\mathrm{sum}} \to U_{\mathrm{sum}} \times s$)

To test the numerical stability and responsiveness of the decision-policy mapping under controlled variations in aggregate uncertainty magnitude, we evaluate scale factors $s \in \{0.80, 0.90, 1.00, 1.10, 1.20\}$ around frozen $\delta^* = 0.10$ and nominal thresholds $(\tau_1=0.20, \tau_2=0.40)$:

| Scale Factor $s$ | Effective $\delta \times s$ | Negative Rate (%) | Policy A Routine (%) | Policy A Additional (%) | Policy A Escalation (%) | Reclassification Rate (%) | Escalation Reduction (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$0.80$** | $0.08$ | $4.4\%$ ($22$) | $43.8\%$ ($219$) | $39.2\%$ ($196$) | $17.0\%$ ($85$) | $15.4\%$ ($77$) | $6.4\%$ ($32$) |
| **$0.90$** | $0.09$ | $6.2\%$ ($31$) | $45.4\%$ ($227$) | $38.0\%$ ($190$) | $16.6\%$ ($83$) | $17.2\%$ ($86$) | $6.8\%$ ($34$) |
| **$1.00$** | **$0.10$** | **$7.4\%$ ($37$)** | **$46.4\%$ ($232$)** | **$37.4\%$ ($187$)** | **$16.2\%$ ($81$)** | **$18.4\%$ ($92$)** | **$7.2\%$ ($36$)** |
| **$1.10$** | $0.11$ | $9.6\%$ ($48$) | $47.8\%$ ($239$) | $36.4\%$ ($182$) | $15.8\%$ ($79$) | $19.6\%$ ($98$) | $7.6\%$ ($38$) |
| **$1.20$** | $0.12$ | $11.6\%$ ($58$) | $49.6\%$ ($248$) | $35.4\%$ ($177$) | $15.0\%$ ($75$) | $21.4\%$ ($107$) | $8.4\%$ ($42$) |

### Diagnostic Findings:
- Reclassification rates increased monotonically across the tested perturbation levels ($15.4\% \to 21.4\%$) as the uncertainty scale factor increased from $0.80$ to $1.20$.
- Negative DCRI rates increased monotonically ($4.4\% \to 11.6\%$) across the tested grid, with all negative scores consistently mapped to Tier 0 (Routine Review).
- These results reflect controlled sensitivity across the pre-specified discrete grid and do not imply that real-world uncertainty was misestimated or that the response is mathematically linear across all possible continuous scales.

---

## 2. Threshold Boundary Jitter Perturbations ($\tau \to \tau \pm \Delta\tau$)

To evaluate sensitivity to operational threshold placement, we apply jitter perturbations $\Delta\tau \in \{-0.02, -0.01, 0.00, +0.01, +0.02\}$ around $(\tau_1=0.20, \tau_2=0.40)$:

| Jitter $\Delta\tau$ | $(\tau_1, \tau_2)$ | Policy A Routine (%) | Policy A Escalation (%) | Policy B Routine (%) | Policy B Escalation (%) | Reclassification Rate (%) | Escalation Reduction (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$-0.02$** | $(0.18, 0.38)$ | $42.2\%$ ($211$) | $18.2\%$ ($91$) | $30.6\%$ ($153$) | $26.8\%$ ($134$) | $18.6\%$ ($93$) | $8.6\%$ ($43$) |
| **$-0.01$** | $(0.19, 0.39)$ | $44.0\%$ ($220$) | $17.6\%$ ($88$) | $33.4\%$ ($167$) | $24.8\%$ ($124$) | $18.2\%$ ($91$) | $7.2\%$ ($36$) |
| **$0.00$** | **$(0.20, 0.40)$** | **$46.4\%$ ($232$)** | **$16.2\%$ ($81$)** | **$35.2\%$ ($176$)** | **$23.4\%$ ($117$)** | **$18.4\%$ ($92$)** | **$7.2\%$ ($36$)** |
| **$+0.01$** | $(0.21, 0.41)$ | $48.2\%$ ($241$) | $15.4\%$ ($77$) | $37.2\%$ ($186$) | $22.2\%$ ($111$) | $18.4\%$ ($92$) | $6.8\%$ ($34$) |
| **$+0.02$** | $(0.22, 0.42)$ | $51.0\%$ ($255$) | $14.4\%$ ($72$) | $39.8\%$ ($199$) | $20.6\%$ ($103$) | $18.4\%$ ($92$) | $6.2\%$ ($31$) |

### Diagnostic Findings (H4 Supported in Benchmark):
- Across the discrete jitter grid evaluated, tier assignments shifted monotonically with threshold offsets.
- The overall reclassification rate remained tightly bounded ($18.2\%\text{--}18.6\%$) across all tested jitter levels on this cohort, demonstrating that the observed reclassifications are driven by the systematic score discount rather than hypersensitivity to exact boundary location.
