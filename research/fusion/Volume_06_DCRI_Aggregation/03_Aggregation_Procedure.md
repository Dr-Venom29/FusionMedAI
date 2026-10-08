# DCRI Aggregation Procedure & Modality Decomposition

## 1. Step-by-Step Computational Workflow

The DCRI aggregation pipeline processes decision packets through 5 deterministic stages:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ Stage 1: Modality Weight & Risk Ingestion                                   │
│   Extract w_i, r_i, U_i for active modalities i in A                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ Stage 2: Weighted Contribution Calculation                                  │
│   Compute K_i = w_i * r_i                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ Stage 3: Risk Aggregation                                                   │
│   Compute R_fusion = sum_{i in A} K_i in [0.0, 1.0]                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ Stage 4: Uncertainty Burden & Penalty Computation                           │
│   Compute U_sum = sum U_i, U_mean, P_U(delta) = delta * U_sum               │
├─────────────────────────────────────────────────────────────────────────────┤
│ Stage 5: Decision Index Synthesis & Verification                            │
│   Compute DCRI = R_fusion - P_U(delta) in [-delta * |A|, 1.0]               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Worked Mathematical Example

Consider a controlled tri-modal decision packet evaluated under full availability $\mathcal{A} = \{\text{retina}, \text{foot}, \text{clinical}\}$:

### Channel Attributes
- **Retina**: $w_R = 0.477$, $r_R = 0.720$, $U_R = 0.100$
- **Foot**: $w_F = 0.268$, $r_F = 0.410$, $U_F = 0.200$
- **Clinical**: $w_C = 0.255$, $r_C = 0.620$, $U_C = 0.150$

### Computations
1. **Weighted Contributions**:
   $$K_R = 0.477 \times 0.720 = 0.34344$$
   $$K_F = 0.268 \times 0.410 = 0.10988$$
   $$K_C = 0.255 \times 0.620 = 0.15810$$

2. **Aggregated Risk**:
   $$R_{\text{fusion}} = 0.34344 + 0.10988 + 0.15810 = 0.61142$$

3. **Uncertainty Burden**:
   $$U_{\text{sum}} = 0.100 + 0.200 + 0.150 = 0.450$$
   $$U_{\text{mean}} = \frac{0.450}{3} = 0.150$$

4. **Uncertainty Penalty ($\delta = 0.20$)**:
   $$p_R = 0.20 \times 0.100 = 0.020$$
   $$p_F = 0.20 \times 0.200 = 0.040$$
   $$p_C = 0.20 \times 0.150 = 0.030$$
   $$P_U(0.20) = 0.020 + 0.040 + 0.030 = 0.090$$

5. **Final Decision Index**:
   $$\text{DCRI}_{0.20} = 0.61142 - 0.090 = 0.52142$$

---

## 3. Decision-Level Interpretability

The exact additive decomposition into $\{K_i\}$ and $\{p_i\}$ provides structured transparency for multimodal decisions:
- **Primary Aggregated-Risk Contribution**: Retina accounts for $56.2\%$ of aggregated risk ($0.343 / 0.611$).
- **Primary Uncertainty Penalty**: Foot accounts for $44.4\%$ of total discount penalty ($0.040 / 0.090$).
- **Net Derived Risk**: Fused risk is revised downwards by $14.7\%$ due to cross-modality predictive uncertainty.
