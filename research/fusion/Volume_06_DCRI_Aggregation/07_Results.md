# Phase C11.6: Empirical Results & Verification Summary

## 1. Primary Empirical Findings Summary

The full Phase C11.6 experimental battery was executed over the frozen $N=500$ controlled decision cohort (seed=115):

```text
================================================================================
FusionMedAI Phase C11.6: DCRI Risk Aggregation & Sensitivity Analysis
Cohort Size: N = 500 | PRNG Seed: 115 | Status: VERIFIED & SEALED
================================================================================
```

### Tri-Modal Baseline Summary ($\mathcal{A} = \{\text{retina}, \text{foot}, \text{clinical}\}$)
- **Mean Fused Risk $R_{\text{fusion}}$**: $0.288499 \pm 0.163198$ (Median: $0.285279$, Range: $[0.011744, 0.793333]$, IQR: $0.251773$)
- **Mean Uncertainty Burden $U_{\text{sum}}$**: $0.633936 \pm 0.144983$
- **Mean Uncertainty Per Modality $U_{\text{mean}}$**: $0.211312 \pm 0.048328$
- **Mean Router Authority Allocation**:
  - Retina: $w_R = 0.477111$ (Dominant in $79.6\%$ of cases)
  - Foot: $w_F = 0.267970$ (Dominant in $10.6\%$ of cases)
  - Clinical: $w_C = 0.254919$ (Dominant in $9.8\%$ of cases)

---

## 2. Multi-Configuration Performance Matrix

| Configuration | Active Count ($M$) | Mean $R_{\text{fusion}}$ | Mean $U_{\text{sum}}$ | Mean DCRI ($\delta=0.0$) | Mean DCRI ($\delta=0.05$) | Mean DCRI ($\delta=0.10$) | Mean DCRI ($\delta=0.20$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Retina Only (R)** | 1 | 0.255037 | 0.081822 | 0.255037 | 0.250946 | 0.246855 | 0.238673 |
| **Foot Only (F)** | 1 | 0.537238 | 0.528302 | 0.537238 | 0.510823 | 0.484408 | 0.431578 |
| **Clinical Only (C)** | 1 | 0.090325 | 0.023812 | 0.090325 | 0.089134 | 0.087944 | 0.085563 |
| **Retina + Foot (RF)** | 2 | 0.344078 | 0.610124 | 0.344078 | 0.313572 | 0.283066 | 0.222053 |
| **Retina + Clinical (RC)** | 2 | 0.203619 | 0.105634 | 0.203619 | 0.198337 | 0.193056 | 0.182492 |
| **Foot + Clinical (FC)** | 2 | 0.316845 | 0.552114 | 0.316845 | 0.289239 | 0.261634 | 0.206422 |
| **Tri-Modal (RFC)** | 3 | **0.288499** | **0.633936** | **0.288499** | **0.256802** | **0.225105** | **0.161712** |
| **Zero Modality** | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 (`NO_MODALITY_AVAILABLE`) |

---

## 3. Synthetic Sanity Cases (Cases A through F)

- **Case A (Identical Inputs, $r=0.6, U=0.1$)**: $R_{\text{fusion}} = 0.600000$ independent of router weights. Passed.
- **Case B (Zero Uncertainty, $U_i=0$)**: $\text{DCRI}_\delta = R_{\text{fusion}}$ for all $\delta \in [0, 1]$. Passed.
- **Case C (Increasing Uncertainty, $U: 0.1 \to 0.2 \to 0.5$)**: $\text{DCRI}_{0.2} = 0.54 \to 0.48 \to 0.30$. Monotonic decrease verified. Passed.
- **Case D ($\delta = 0.0$)**: $\text{DCRI}_0 = R_{\text{fusion}}$ down to float precision. Passed.
- **Case E (Unimodal $A_R=1$)**: $w_R = 1.0, R_{\text{fusion}} = 0.75, \text{DCRI}_{0.2} = 0.71$. Passed.
- **Case F (Zero Modalities)**: State emitted as `NO_MODALITY_AVAILABLE` with $0.0$ values. Passed.

---

## 4. Edge Cases & Numerical Robustness

- **Unclamped Negative DCRI**: Packet with $r_i=0.10, U_i=0.80, \delta=1.0$ evaluated to $\text{DCRI} = -2.300000 \in [-3.0, 1.0]$. Verified.
- **Extreme Risk Boundaries under Zero Uncertainty**: $r=0.0, U=0.0 \implies \text{DCRI}=0.0$, and $r=1.0, U=0.0, \delta=0.0 \implies \text{DCRI}=1.0$. Verified.
- **NaN / Inf Rejection**: Validated at contract boundaries. Passed.
- **Deterministic Repeatability**: Identical float results across independent passes ($f(X) \equiv f(X)$). Passed.
