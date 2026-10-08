# Uncertainty vs Cross-Modality Conflict Analysis

## 1. Orthogonality Between Conflict and Predictive Uncertainty

A central methodological objective of Phase C11.7 is evaluating whether cross-modality risk disagreement is distinct from individual modality predictive uncertainty:

$$\text{Correlation}(\Delta_{\max}, U_{\text{sum}}) = 0.088166$$

The linear correlation is near zero ($r \approx 0.088$), confirming that **cross-modality discordance and predictive uncertainty represent largely orthogonal dimensions**.

---

## 2. Uncertainty Burden Stratified by Conflict Severity

| Conflict Severity Band | Packet Count | Rate | Mean $\Delta_{\max}$ | Mean $U_{\text{sum}}$ | Std $U_{\text{sum}}$ | Mean DCRI ($\delta=0.20$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LOW ($\Delta_{\max} < 0.20$)** | 45 | 9.0% | 0.124578 | 0.587932 | 0.143684 | 0.140228 |
| **MODERATE ($0.20 \le \Delta_{\max} < 0.35$)** | 92 | 18.4% | 0.280459 | 0.627254 | 0.156108 | 0.145892 |
| **HIGH ($\Delta_{\max} \ge 0.35$)** | 363 | 72.6% | 0.630252 | 0.641340 | 0.141269 | 0.168393 |

---

## 3. Methodological Implications

1. **High Disagreement with Low Uncertainty**: Decision packets can exhibit high risk divergence ($\Delta_{\max} \ge 0.50$) while individual models report high predictive confidence ($U_i \le 0.10$).
2. **High Uncertainty with Consensus**: Models may agree on intermediate risk ($r \approx 0.30$) while reporting elevated uncertainty ($U \ge 0.40$).
3. **Separate Diagnostic Value**: Treating discordance as a separate metric prevents the false assumption that confident models never conflict.
