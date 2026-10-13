# Chapter 06: DCRI Propagation and Discount Ablation

## 1. Formulation of Downstream Uncertainty Discounting

The Dynamic Composite Risk Index (DCRI) modifies raw fused risk $R_{\text{fusion}}$ by subtracting an epistemic uncertainty penalty $P_U$:

$$\text{DCRI} = R_{\text{fusion}} - P_U = R_{\text{fusion}} - \delta^* \sum_{i \in \mathcal{A}} U_i$$

where $\delta^* = 0.10$ is the prespecified discount factor. Under the canonical Phase C11.6/C11.13 specification, the uncertainty penalty $P_U$ is computed as the unweighted sum of active modality epistemic uncertainties ($U_{\text{sum}} = \sum_{i \in \mathcal{A}} U_i$). Because this penalty depends strictly on the set of available channels $\mathcal{A}$ rather than the dynamic routing weights $w_i$, the uncertainty penalty remains identical across all routing variants evaluating the same inputs ($\Delta P_U = 0.0$).

The uncertainty penalty acts as a risk buffer or safety margin: when input modalities carry high epistemic uncertainty ($U_i$), the composite index is penalized downward to reflect diminished evidential confidence.

In Variant $A5$, the discount is completely ablated by setting $\delta = 0.0$:
$$\text{DCRI}^{(A5)} = R_{\text{fusion}}^{(A5)} - 0.0 = R_{\text{fusion}}^{(A0)}$$

This provides a direct experimental measurement of the impact of the epistemic uncertainty buffer on overall risk levels and decision triage.

---

## 2. Quantitative Impact of Discount Ablation (Variant A5)

### 2.1 Score Shift and Uncertainty Buffer Magnitude
In Variant $A5$, the difference in DCRI for each packet $k$ is identically equal to the reference uncertainty penalty:
$$\Delta \text{DCRI}(k) = \text{DCRI}^{(A5)}(k) - \text{DCRI}^{(A0)}(k) = P_U^{(A0)}(k) = \delta^* \sum_{i \in \mathcal{A}} U_i(k)$$

| Metric | Reference B6 ($A0$) | Ablated ($\delta=0, A5$) | Paired Shift $\Delta$ | $95\%$ Bootstrap CI |
| :--- | :---: | :---: | :---: | :---: |
| **Mean DCRI $\overline{\text{DCRI}}$** | $0.197624$ | $0.261017$ | $+0.063394$ | $[+0.061150, +0.065409]$ |
| **Median DCRI** | $0.150263$ | $0.218151$ | $+0.064112$ | $[+0.062018, +0.066114]$ |
| **Std Dev $\sigma_{\text{DCRI}}$** | $0.172243$ | $0.163198$ | $-0.009045$ | $[-0.010500, -0.007600]$ |
| **Negative DCRI Count** | $38$ ($7.6\%$) | $0$ ($0.0\%$) | $-38$ | $[-42, -31]$ |

### 2.2 Complete Elimination of Negative DCRI Values
In reference architecture $A0$, when raw fused risk is very low ($R_{\text{fusion}} < 0.06$) and uncertainty is moderate ($U_i > 0.50$), the subtraction $R_{\text{fusion}} - 0.10 \sum U_i$ produces small negative composite values (minimum observed in canonical $A0$ records: $-0.028203$).
- In $A0$, exactly $38/500$ packets ($7.6\%$) exhibit negative DCRI values.
- In $A5$ ($\delta=0$), all DCRI values are identical to $R_{\text{fusion}} \ge 0.000000$. Exactly **0** packets exhibit negative values.
- This demonstrates that negative DCRI values are entirely an artifact of linear subtraction without non-negativity clipping $\max(0, R - P_U)$.


---

## 3. Comparative DCRI Metrics Across All Ablations

The table below contrasts the mean DCRI, paired differences, and negative DCRI counts across all candidate variants:

| Variant | Target Component | Mean DCRI $\overline{\text{DCRI}}$ | Paired $\overline{\Delta \text{DCRI}}$ | $95\%$ Bootstrap CI | Negative DCRI Count | Negative Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **A0** | Reference Architecture | $0.197624$ | $0.000000$ | — | $38$ | $7.6\%$ |
| **A1** | No Confidence ($\alpha=0$) | $0.211368$ | $+0.013744$ | $[+0.011245, +0.016254]$ | $34$ | $6.8\%$ |
| **A2** | No Reliability ($\beta=0$) | $0.189798$ | $-0.007825$ | $[-0.008456, -0.007194]$ | $41$ | $8.2\%$ |
| **A3** | No Uncertainty Router ($\gamma=0$) | $0.227961$ | $+0.030338$ | $[+0.027210, +0.033466]$ | $24$ | $4.8\%$ |
| **A4** | No Quality ($\eta=0$) | $0.196455$ | $-0.001168$ | $[-0.001422, -0.000914]$ | $38$ | $7.6\%$ |
| **A5** | No DCRI Discount ($\delta=0$) | $0.261017$ | $+0.063394$ | $[+0.061150, +0.065409]$ | $0$ | $0.0\%$ |
| **A_UNIFORM** | Uniform Baseline | $0.233410$ | $+0.035787$ | $[+0.030412, +0.041162]$ | $34$ | $6.8\%$ |

### Key Findings:
1. **$A5$ Has the Largest Upward DCRI Impact:** Removing the uncertainty discount shifts scores upward by $+0.063394$, more than twice the magnitude of any router-level modification, eliminating all negative scores.
2. **$A3$ Also Increases DCRI:** Removing the router uncertainty penalty increases DCRI by $+0.030338$, purely because higher-risk foot ulcer predictions are no longer suppressed.
3. **$A1$ Increases DCRI:** Removing dynamic confidence increases DCRI by $+0.013744$, as authority shifts away from the lower-risk clinical tabular channel toward higher-risk vision channels.
4. **Orthogonality Certified (Gate C16-06):** In $A5$, $\Delta w_i \equiv 0.0$ and $\Delta R_{\text{fusion}} \equiv 0.0$. The downstream discount is mathematically and functionally orthogonal to router weighting.

