# Uncertainty Penalty & Double-Use Analysis

## 1. Cumulative Burden ($\sum U_i$) vs Mean Uncertainty ($\frac{1}{|\mathcal{A}|}\sum U_i$)

The current mathematical definition of DCRI applies a penalty based on cumulative uncertainty:
$$P_U(\delta) = \delta \sum_{i \in \mathcal{A}} U_i$$

### Property of Cumulative Penalty
For a fixed decision packet, adding an available modality with $U_{\text{new}} \ge 0$ cannot decrease $U_{\text{sum}}$ ($U_{\text{sum}}' = U_{\text{sum}} + U_{\text{new}} \ge U_{\text{sum}}$).

| Configuration | Active Channels | Mean $U_{\text{sum}}$ | Mean $U_{\text{mean}}$ | Mean Penalty ($\delta=0.20$) |
| :--- | :--- | :--- | :--- | :--- |
| **Unimodal Retina (R)** | $\{\text{retina}\}$ | 0.0818 | 0.0818 | 0.0164 |
| **Unimodal Foot (F)** | $\{\text{foot}\}$ | 0.5283 | 0.5283 | 0.1057 |
| **Unimodal Clinical (C)** | $\{\text{clinical}\}$ | 0.0238 | 0.0238 | 0.0048 |
| **Bimodal Retina + Foot (RF)** | $\{\text{retina}, \text{foot}\}$ | 0.6101 | 0.3051 | 0.1220 |
| **Bimodal Retina + Clinical (RC)** | $\{\text{retina}, \text{clinical}\}$ | 0.1056 | 0.0528 | 0.0211 |
| **Bimodal Foot + Clinical (FC)** | $\{\text{foot}, \text{clinical}\}$ | 0.5521 | 0.2761 | 0.1104 |
| **Tri-Modal (RFC)** | $\{\text{retina}, \text{foot}, \text{clinical}\}$ | **0.6339** | **0.2113** | **0.1268** |

### Insight
Adding a modality always increases $U_{\text{sum}}$ by $U_{\text{new}} \ge 0$. Adding a low-uncertainty channel (such as Clinical, $U_C \approx 0.024$) introduces only a negligible penalty increase ($+0.0048$ at $\delta=0.2$), whereas adding Foot introduces a substantial penalty ($+0.1057$).

---

## 2. Investigation of Uncertainty "Double-Use"

Uncertainty is incorporated into the pipeline at two distinct computational tiers:

1. **Tier 1 (ACARA-U Dynamic Routing)**:
   $$z_i = \alpha C_i + \beta R_i - \gamma U_i + \eta Q_i$$
   High $U_i$ decreases logit $z_i$, reducing modality weight $w_i$.
2. **Tier 2 (DCRI Risk Aggregation)**:
   $$\text{DCRI} = R_{\text{fusion}} - \delta \sum_{i \in \mathcal{A}} U_i$$
   High $U_i$ subtracts an explicit scalar penalty from the aggregated risk index.

### Empirical Comparison: Regime A vs Regime B

```text
Regime A (Routing Uncertainty Only, δ=0.0):
  Mean R_fusion = 0.288499
  Mean DCRI     = 0.288499
  Penalty       = 0.000000

Regime B (Routing + DCRI Discount, δ=0.20):
  Mean R_fusion = 0.288499
  Mean DCRI     = 0.161712
  Penalty       = 0.126787  (-43.9% relative index discount)
```

### Scientific Conclusion on Double-Use
- **Non-Redundant Roles**: The two uses of uncertainty perform distinct mathematical operations:
  1. *Router Weighting* governs **relative authority** among active modalities (modulating the balance between high-certainty and low-certainty channels).
  2. *DCRI Penalty* provides an **absolute additive discount**, systematically shifting the aggregate index to enforce decision-level risk aversion.
- Phase C11.12 will empirically evaluate whether $\delta=0$ or $\delta > 0$ provides superior decision-level performance under a pre-specified validation criterion.
