# Volume 13 — DCRI Global Uncertainty Penalty Selection ($\delta \in [0.0, 1.0]$)

> **Phase C11.13 Research Documentation & Parameter Freeze Report**  
> **Status:** 🟢 SEALED & VERIFIED (20/20 Gates Passed)  
> **Selected Parameter:** $\delta^* = 0.10$ (Conservative Uncertainty Discounting)  
> **Evaluation Cohort:** Frozen C11 Controlled Multi-Source Cohort ($N=500$, Seed 115)  
> **Router State:** Frozen Reference Configuration $\Theta_0 = (\alpha=1.0, \beta=1.5, \gamma=1.0, \eta=0.5)$

---

## Executive Summary

Phase C11.13 addresses the formal empirical selection of the global uncertainty penalty multiplier $\delta$ in the Decision Confidence & Risk Index ($\text{DCRI}_\delta$):

$$\text{DCRI}_\delta = R_{\text{fusion}} - \delta \sum_{i \in \mathcal{A}} U_i$$

Prior to this phase, $\delta = 0.20$ served as a historical provisional operating point. Phase C11.13 evaluates a pre-specified 11-point candidate grid spanning $\delta \in \{0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00\}$ across the frozen $N=500$ controlled decision cohort under strictly frozen upstream routing weights, calibrated modality risks, and predictive uncertainties.

Through a multi-tiered selection hierarchy enforcing mathematical invariants, behavioral feasibility boundaries, modality availability consistency, and parsimonious uncertainty attenuation, **$\delta^* = 0.10$** is formally selected and frozen as the standard operating parameter.

---

## Key Empirical Findings

1. **Uncertainty Penalty Scaling**: Mean summed predictive uncertainty across active modalities is $\overline{U_{\text{sum}}} = 0.633936 \pm 0.154212$. The analytical derivative $\frac{\partial \overline{\text{DCRI}}}{\partial \delta} = -\overline{U_{\text{sum}}} = -0.633936$ holds with exact zero residual ($< 10^{-16}$) across all candidate intervals.
2. **Behavioral Contrast ($\delta^* = 0.10$ vs Historical $\delta = 0.20$)**:
   - At $\delta^* = 0.10$, the mean uncertainty penalty is $\overline{P} = 0.063394$ ($24.29\%$ of base fused risk $\overline{R_{\text{fusion}}} = 0.261017$), yielding a mean $\text{DCRI} = 0.197624$ with a restrained negative-DCRI rate of **$7.6\%$** (38/500 packets).
   - In contrast, historical provisional $\delta = 0.20$ imposed a heavy penalty of $\overline{P} = 0.126787$ ($48.57\%$ of base risk), forcing **$25.4\%$** (127/500 packets) into negative DCRI territory.
3. **Rank Stability**: $\delta^* = 0.10$ preserves near-perfect rank alignment with the underlying fused risk ranking (Spearman $\rho_s = 0.9872$), whereas aggressive penalties ($\delta \ge 0.50$) induce severe rank distortion ($\rho_s \le 0.8257$).
4. **Availability Invariance Across Regimes**: Negative DCRI rates scale smoothly with modality cardinality: at $\delta^* = 0.10$, single-modality regimes ($M=1$) produce $0.0\%$ to $2.2\%$ (`R`: 0.0%, `C`: 0.0%, `F`: 2.2%), dual-modality regimes ($M=2$) produce $4.4\%$ to $8.2\%$, and full triple modality ($M=3$, `RFC`) produces $7.6\%$ (38/500 packets).

---

## Master Candidate Evaluation Scoreboard

| Candidate ID | $\delta$ Value | Category | Mean DCRI | Negative Rate | Mean Penalty $\overline{P_\delta}$ | Penalty / Base Ratio | Spearman $\rho_s$ | Status |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **D00** | $0.00$ | Zero Penalty | $0.261017$ | $0.0\%$ (0/500) | $0.000000$ | $0.00\%$ | $1.0000$ | Feasible Baseline |
| **D05** | $0.05$ | Conservative | $0.229320$ | $0.0\%$ (0/500) | $0.031697$ | $12.14\%$ | $0.9966$ | Feasible |
| **D10** | **$0.10$** | **Selected** | **$0.197624$** | **$7.6\%$ (38/500)** | **$0.063394$** | **$24.29\%$** | **$0.9872$** | 🟢 **SELECTED ($\delta^*$)** |
| **D15** | $0.15$ | Conservative | $0.165927$ | $18.6\%$ (93/500) | $0.095090$ | $36.43\%$ | $0.9728$ | Feasible |
| **D20** | $0.20$ | Historical Prov. | $0.134230$ | $25.4\%$ (127/500) | $0.126787$ | $48.57\%$ | $0.9560$ | Feasible (High Neg.) |
| **D25** | $0.25$ | Moderate | $0.102533$ | $30.8\%$ (154/500) | $0.158484$ | $60.72\%$ | $0.9367$ | Feasible Boundary |
| **D30** | $0.30$ | Moderate | $0.070836$ | $39.2\%$ (196/500) | $0.190181$ | $72.86\%$ | $0.9161$ | Rejected (Excess Pen.) |
| **D40** | $0.40$ | Substantial | $0.007443$ | $55.6\%$ (278/500) | $0.253574$ | $97.15\%$ | $0.8713$ | Rejected (Pathological) |
| **D50** | $0.50$ | Substantial | $-0.055951$ | $67.4\%$ (337/500) | $0.316968$ | $121.44\%$ | $0.8257$ | Rejected (Scale Invert) |
| **D75** | $0.75$ | Boundary | $-0.214435$ | $79.6\%$ (398/500) | $0.475452$ | $182.15\%$ | $0.7255$ | Rejected (Boundary) |
| **D100** | $1.00$ | Boundary | $-0.372919$ | $84.2\%$ (421/500) | $0.633936$ | $242.87\%$ | $0.6488$ | Rejected (Boundary) |

---

## Volume Documentation Index

- [Chapter 01 — Scientific Protocol & Research Hypotheses](01_Protocol.md)
- [Chapter 02 — Candidate Grid Specification & Domain Analysis](02_Candidate_Grid.md)
- [Chapter 03 — Full Empirical Cohort Results](03_Results.md) (Includes Figures 1 & 2)
- [Chapter 04 — Regime Stratification & Bootstrap Statistics](04_Statistical_Analysis.md) (Includes Figures 3 & 4)
- [Chapter 05 — Selection Hierarchy & Rationale](05_Selection_Rationale.md)
- [Chapter 06 — Methodological Boundaries & Limitations](06_Limitations.md)
- [Chapter 07 — Freeze Report & Verification Audit](07_Freeze_Report.md)
