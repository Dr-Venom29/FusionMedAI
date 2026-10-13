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
- **Mean Fused Risk $R_{\text{fusion}}$**: $0.259623$ ($\Theta_{\text{unit}}$) / $0.261017$ ($\Theta_0$)
- **Mean Uncertainty Burden $U_{\text{sum}}$**: $0.633936 \pm 0.144983$ ($\overline{P}_{U,0.10} = 0.063394$, $\overline{P}_{U,0.20} = 0.126787$)
- **Mean Uncertainty Per Modality $U_{\text{mean}}$**: $0.211312 \pm 0.048328$
- **Mean Router Authority Allocation ($\Theta_{\text{unit}} / \Theta_0$)**:
  - Retina: $w_R = 0.428717 / 0.427135$ (Dominant in $73.4\% / 73.2\%$ of cases)
  - Foot: $w_F = 0.219386 / 0.223079$ (Dominant in $4.6\% / 4.8\%$ of cases)
  - Clinical: $w_C = 0.351897 / 0.349785$ (Dominant in $22.0\% / 22.0\%$ of cases)

---

## 2. Multi-Configuration Performance Matrix & $\delta$-Sweep ($N=500$)

| Configuration | Active Count ($M$) | Mean $R_{\text{fusion}}$ ($\Theta_0$) | Mean $U_{\text{sum}}$ | Mean DCRI ($\delta=0.0$) | Mean DCRI ($\delta=0.05$) | Mean DCRI ($\delta=0.10$) | Mean DCRI ($\delta=0.20$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Retina Only (R)** | 1 | 0.271421 | 0.165213 | 0.271421 | 0.263160 | 0.254899 | 0.238378 |
| **Foot Only (F)** | 1 | 0.499635 | 0.371204 | 0.499635 | 0.481075 | 0.462515 | 0.425394 |
| **Clinical Only (C)** | 1 | 0.119355 | 0.097518 | 0.119355 | 0.114480 | 0.109604 | 0.099852 |
| **Retina + Foot (RF)** | 2 | 0.341108 | 0.536418 | 0.341108 | 0.314287 | 0.287466 | 0.233824 |
| **Retina + Clinical (RC)** | 2 | 0.200193 | 0.262732 | 0.200193 | 0.187057 | 0.173920 | 0.147647 |
| **Foot + Clinical (FC)** | 2 | 0.263029 | 0.468723 | 0.263029 | 0.239593 | 0.216157 | 0.169285 |
| **Tri-Modal (RFC, $\Theta_0$)** | 3 | **0.261017** | **0.633936** | **0.261017** ($0$ neg) | **0.229320** ($0$ neg) | **0.197624** ($38$ neg) | **0.134230** ($127$ neg) |
| **Tri-Modal (RFC, $\Theta_{\text{unit}}$)** | 3 | **0.259623** | **0.633936** | **0.259623** ($0$ neg) | **0.227926** ($0$ neg) | **0.196229** ($39$ neg) | **0.132835** ($128$ neg) |
| **Zero Modality** | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 (`NO_MODALITY_AVAILABLE`) |

*Note: At $\delta = 0.50$ and $\delta = 1.00$, Tri-Modal ($\Theta_0 / \Theta_{\text{unit}}$) yields $\overline{\mathrm{DCRI}}_{0.50} = -0.055951 / -0.057345$ ($337$ negative packets) and $\overline{\mathrm{DCRI}}_{1.00} = -0.372919 / -0.374314$ ($421$ negative packets).*

---

## 3. Synthetic Sanity Cases (Cases A through F)

- **Case A (Identical Inputs, $r=0.6, U=0.1$)**: $R_{\text{fusion}} = 0.600000$ independent of router weights. Passed.
- **Case B (Zero Uncertainty, $U_i=0$)**: $\text{DCRI}_\delta = R_{\text{fusion}}$ for all $\delta \in [0, 1]$. Passed.
- **Case C (Increasing Uncertainty, $U: 0.1 \to 0.2 \to 0.5$)**: $\text{DCRI}_{0.2} = 0.54 \to 0.48 \to 0.30$. Monotonic decrease verified. Passed.
- **Case D ($\delta = 0.0$)**: $\text{DCRI}_0 = R_{\text{fusion}}$ down to float precision. Passed.
- **Case E (Unimodal $A_R=1$)**: $w_R = 1.0, R_{\text{fusion}} = 0.75, \text{DCRI}_{0.2} = 0.71$, with inactive `foot` and `clinical` entries strictly zeroed ($0.0$) across all 5 channel maps (`modality_weights`, `modality_risks`, `modality_uncertainties`, `weighted_risk_contributions`, `uncertainty_penalty_contributions`). Passed.
- **Case F (Zero Modalities)**: State emitted as `NO_MODALITY_AVAILABLE` with $0.0$ values. Passed.

---

## 4. Edge Cases & Numerical Robustness

- **Unclamped Negative DCRI**: Packet with $r_i=0.10, U_i=0.80, \delta=1.0$ evaluated to $\text{DCRI} = -2.300000 \in [-3.0, 1.0]$. Verified.
- **Extreme Risk Boundaries under Zero Uncertainty**: $r=0.0, U=0.0 \implies \text{DCRI}=0.0$, and $r=1.0, U=0.0, \delta=0.0 \implies \text{DCRI}=1.0$. Verified.
- **NaN / Inf Rejection**: Validated at contract boundaries. Passed.
- **Deterministic Repeatability**: Identical float results across independent passes ($f(X) \equiv f(X)$). Passed.

