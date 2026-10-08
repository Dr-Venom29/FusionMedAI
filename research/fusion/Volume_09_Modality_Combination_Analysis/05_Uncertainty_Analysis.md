# Modality Uncertainty & DCRI Scaling Analysis

## 1. Uncertainty Scaling Across Modality Cardinality

In the Dual-Constraint Risk Index (DCRI), total predictive uncertainty is the sum of available channel uncertainties:

$$
U_{\text{sum}} = \sum_{i \in \mathcal{A}} U_i
$$

Unavailable channels have $A_i = 0 \implies U_i \notin \mathcal{A}$, contributing $0.0$ to $U_{\text{sum}}$.

### Empirical Uncertainty by Combination (D2 Cohort)

| Combination | Modality Channels Active | Mean $U_{\text{sum}}$ | Std $U_{\text{sum}}$ | Min $U_{\text{sum}}$ | Max $U_{\text{sum}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RFC** | Retina + Foot + Clinical | **0.6728** | $0.2312$ | $0.2104$ | $1.4120$ |
| **RF** | Retina + Foot | **0.5456** | $0.2530$ | $0.1420$ | $1.2850$ |
| **RC** | Retina + Clinical | **0.3241** | $0.2985$ | $0.0780$ | $1.0250$ |
| **FC** | Foot + Clinical | **0.2644** | $0.3951$ | $0.0650$ | $1.1540$ |
| **R** | Retina Only | **0.2825** | $0.3341$ | $0.0350$ | $0.9850$ |
| **F** | Foot Only | **0.2114** | $0.3012$ | $0.0420$ | $0.9200$ |
| **C** | Clinical Only | **0.0985** | $0.1845$ | $0.0210$ | $0.4850$ |

---

## 2. DCRI Penalty Interaction Across Frequency Tiers

The Dual-Constraint Risk Index aggregates fused risk with an uncertainty penalty $\delta = 0.20$:

$$
\text{DCRI}_{\delta=0.20} = R_{\text{fusion}} - 0.20 \cdot U_{\text{sum}}
$$

### Tier-Level Uncertainty and DCRI Dynamics (D2 Moderate Tail)

- **HEAD Tier (RFC, RF)**:
  - $\overline{U_{\text{sum}}} = 0.6198 \pm 0.2487$
  - Uncertainty Penalty: $-0.20 \times 0.6198 = -0.1240$
  - Result: $\overline{\text{DCRI}} = 0.1930 \pm 0.2095$ (relative to $\overline{R_{\text{fusion}}} = 0.3169$)
- **MIDDLE Tier (RC, FC)**:
  - $\overline{U_{\text{sum}}} = 0.3002 \pm 0.3412$
  - Uncertainty Penalty: $-0.20 \times 0.3002 = -0.0600$
  - Result: $\overline{\text{DCRI}} = 0.1796 \pm 0.1669$ (relative to $\overline{R_{\text{fusion}}} = 0.2397$)
- **TAIL Tier (R, F, C)**:
  - $\overline{U_{\text{sum}}} = 0.2097 \pm 0.3039$
  - Uncertainty Penalty: $-0.20 \times 0.2097 = -0.0419$
  - Result: $\overline{\text{DCRI}} = 0.2305 \pm 0.2453$ (relative to $\overline{R_{\text{fusion}}} = 0.2725$)

### Key Takeaway
Because $U_{\text{sum}}$ is additive across available modalities, multi-modal configurations can accumulate larger total uncertainty than unimodal configurations. The observed cohort means show a general cardinality-related increase, but not strict monotonic ordering across every individual combination (e.g., $U_{\text{sum}}^{\text{R}} = 0.2825 > U_{\text{sum}}^{\text{FC}} = 0.2644$). Under the current DCRI formulation, tri-modal packets incur a larger absolute uncertainty penalty when their summed uncertainty is larger.
