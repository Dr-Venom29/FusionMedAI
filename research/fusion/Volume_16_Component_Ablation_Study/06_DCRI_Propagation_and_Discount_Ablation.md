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
| **Mean DCRI $\overline{\text{DCRI}}$** | $0.226506$ | $0.289900$ | $+0.063394$ | $[+0.061150, +0.065409]$ |
| **Median DCRI** | $0.181244$ | $0.245812$ | $+0.064112$ | $[+0.062018, +0.066114]$ |
| **Std Dev $\sigma_{\text{DCRI}}$** | $0.182104$ | $0.178491$ | $-0.003613$ | $[-0.004921, -0.002280]$ |
| **Negative DCRI Count** | $37$ ($7.4\%$) | $0$ ($0.0\%$) | $-37$ | $[-42, -31]$ |

### 2.2 Complete Elimination of Negative DCRI Values
In reference architecture $A0$, when raw fused risk is very low ($R_{\text{fusion}} < 0.06$) and uncertainty is moderate ($U_i > 0.50$), the subtraction $R_{\text{fusion}} - 0.10 \sum U_i$ produces small negative composite values (minimum observed in canonical $A0$ records: $-0.027143$; mean of negative scores: $-0.010750$).
- In $A0$, exactly $37/500$ packets ($7.4\%$) exhibit negative DCRI values.
- In $A5$ ($\delta=0$), all DCRI values are identical to $R_{\text{fusion}} \ge 0.000000$. Exactly **0** packets exhibit negative values.
- This demonstrates that negative DCRI values are entirely an artifact of linear subtraction without non-negativity clipping $\max(0, R - P_U)$.

---

## 3. Comparative DCRI Metrics Across All Ablations

The table below contrasts the mean DCRI, paired differences, and negative DCRI counts across all candidate variants:

| Variant | Target Component | Mean DCRI $\overline{\text{DCRI}}$ | Paired $\overline{\Delta \text{DCRI}}$ | $95\%$ Bootstrap CI | Negative DCRI Count | Negative Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **A0** | Reference Architecture | $0.226506$ | $0.000000$ | — | $37$ | $7.4\%$ |
| **A1** | No Confidence ($\alpha=0$) | $0.211367$ | $-0.015139$ | $[-0.018219, -0.011936]$ | $44$ | $8.8\%$ |
| **A2** | No Reliability ($\beta=0$) | $0.220313$ | $-0.006193$ | $[-0.006796, -0.005625]$ | $38$ | $7.6\%$ |
| **A3** | No Uncertainty Router ($\gamma=0$) | $0.257348$ | $+0.030842$ | $[+0.027805, +0.033985]$ | $29$ | $5.8\%$ |
| **A4** | No Quality ($\eta=0$) | $0.225914$ | $-0.000592$ | $[-0.000846, -0.000360]$ | $37$ | $7.4\%$ |
| **A5** | No DCRI Discount ($\delta=0$) | $0.289900$ | $+0.063394$ | $[+0.061150, +0.065409]$ | $0$ | $0.0\%$ |
| **A_UNIFORM** | Uniform Baseline | $0.233410$ | $+0.006904$ | $[+0.001625, +0.012249]$ | $34$ | $6.8\%$ |

### Key Findings:
1. **$A5$ Has the Largest Upward DCRI Impact:** Removing the uncertainty discount shifts scores upward by $+0.063394$, more than twice the magnitude of any router-level modification.
2. **$A3$ Also Increases DCRI:** Removing the router uncertainty penalty increases DCRI by $+0.030842$, purely because higher-risk foot ulcer predictions are no longer suppressed.
3. **$A1$ Lowers DCRI:** Removing dynamic confidence reduces DCRI by $-0.015139$, increasing the number of negative scores to 44.
4. **Orthogonality Certified (Gate C16-06):** In $A5$, $\Delta w_i \equiv 0.0$ and $\Delta R_{\text{fusion}} \equiv 0.0$. The downstream discount is mathematically and functionally orthogonal to router weighting.
