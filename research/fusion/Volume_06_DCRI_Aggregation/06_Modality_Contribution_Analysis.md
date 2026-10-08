# Modality Contribution & Authority Allocation Analysis

## 1. Weighted Risk Contribution Breakdown ($K_i = w_i r_i$)

Across the $N=500$ controlled decision packets evaluated under full tri-modal availability $\mathcal{A} = \{\text{retina}, \text{foot}, \text{clinical}\}$:

| Modality | Mean Weight $\overline{w_i}$ | Mean Risk $\overline{r_i}$ | Mean Weighted Contribution $\overline{K_i}$ | Relative Risk Share ($\overline{K_i} / \overline{R_{\text{fusion}}}$) | Mean Uncertainty $\overline{U_i}$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Retina** | **0.477111** (47.7%) | 0.255037 | **0.121681** | **42.2%** | 0.081822 |
| **Foot** | **0.267970** (26.8%) | 0.537238 | **0.143965** | **49.9%** | 0.528302 |
| **Clinical** | **0.254919** (25.5%) | 0.090325 | **0.022853** | **7.9%** | 0.023812 |
| **Total / Fused** | **1.000000** | — | **0.288499** | **100.0%** | **0.633936** |

---

## 2. Risk Contribution Dynamics

1. **Authority vs Risk Magnitude**:
   - Retina receives the highest authority weight ($47.7\%$) driven by superior reliability prior ($R_R=0.929956$) and low uncertainty ($U_R=0.0818$).
   - Foot receives moderate authority ($26.8\%$) but contributes the largest portion of absolute fused risk ($49.9\%$) because of higher mean projected risk in the Foot modality cohort ($\overline{r_F} = 0.5372$).
   - Clinical receives $25.5\%$ authority weight but accounts for only $7.9\%$ of aggregated risk because its mean projected readmission risk in the frozen clinical validation cohort is $0.0903$.

2. **Conservation Invariant Audit**:
   $$\sum_{i} \overline{K_i} = 0.121681 + 0.143965 + 0.022853 = 0.288499 = \overline{R_{\text{fusion}}}$$
   Error: $|0.288499 - 0.288499| = 0.000000$. Exact conservation confirmed.

---

## 3. Behavior Across the Seven Availability Regimes

```text
                                Mean Fused Risk R_fusion
Unimodal:
  Retina (R)          [0.2550] ──────────────────────┐
  Foot (F)            [0.5372] ─────────────────────────────────────────────┐
  Clinical (C)        [0.0903] ────────┐                                    │
                                                                            │
Bimodal:                                                                    │
  Retina + Foot (RF)  [0.3441] ───────────────────────────────┐             │
  Retina + Clin (RC)  [0.2036] ──────────────────┐            │             │
  Foot + Clin (FC)    [0.3168] ───────────────────────────┐   │             │
                                                          │   │             │
Tri-Modal:                                                │   │             │
  Retina+Foot+Clin    [0.2885] ─────────────────────────────┐ │             │
                               │  0.1  0.2  0.3  0.4  0.5  0.6
```

- When only one modality is present, authority is strictly $w_i = 1.0$ and $R_{\text{fusion}} = r_i$.
- Multimodal aggregation provides smooth interpolation anchored by router confidence, reliability, quality, and uncertainty.
