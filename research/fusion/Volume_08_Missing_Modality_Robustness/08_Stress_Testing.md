# Behavioral Stress Testing & Masked-Value Invariance

## 1. Targeted Stress Dropout Scenarios

To evaluate system behavior under non-random, targeted information-loss scenarios, 7 pre-specified behavioral stress dropouts were executed over all $N=500$ packets:

| Stress Scenario | Selection Criterion | Primary Modality Dropped | Mean $\Delta R$ | Mean $\Delta \text{DCRI}_{0.20}$ | 95% Bootstrap CI ($\Delta R$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`missing_highest_confidence`** | $\arg\max_{i} C_i$ | Retina ($74.6\%$) / Foot ($25.4\%$) | **0.134542** | 0.140280 | $[0.126410, 0.142850]$ |
| **`missing_lowest_confidence`** | $\arg\min_{i} C_i$ | Clinical ($82.0\%$) / Foot ($18.0\%$) | **0.068412** | 0.074210 | $[0.063200, 0.073800]$ |
| **`missing_highest_reliability`** | $\arg\max_{i} R_i$ | Retina ($100.0\%$) | **0.145687** | 0.145728 | $[0.137358, 0.154205]$ |
| **`missing_lowest_uncertainty`** | $\arg\min_{i} U_i$ | Retina ($98.0\%$) / Clinical ($2.0\%$) | **0.144820** | 0.145100 | $[0.136500, 0.153100]$ |
| **`missing_highest_uncertainty`**| $\arg\max_{i} U_i$ | Foot ($100.0\%$) | **0.103242** | 0.123969 | $[0.095084, 0.111734]$ |
| **`missing_lowest_quality`** | $\arg\min_{i} Q_i$ | Clinical ($96.4\%$) / Foot ($3.6\%$) | **0.063410** | 0.069120 | $[0.058800, 0.068100]$ |
| **`missing_highest_quality`** | $\arg\max_{i} Q_i$ | Retina ($95.2\%$) / Foot ($4.8\%$) | **0.143210** | 0.144100 | $[0.135100, 0.151400]$ |

---

## 2. Masked-Value Invariance Certification

### Masked-Value Invariance Property:
When a modality is marked unavailable ($A_i = 0$), no internal scalar values within that channel (risk $r_i$, confidence $C_i$, uncertainty $U_i$, quality $Q_i$, reliability $R_i$) may influence active authority weights $w_{\text{active}}$, fused risk $R_{\text{fusion}}$, or $\text{DCRI}$.

### Experimental Verification ($N=500$ packets $\times$ 3 bimodal pairs $\times$ 5 extreme perturbation sets = 7,500 trials):
- **Total Invariance Checks**: $7,500$
- **Passed Invariance Checks**: **$7,500$ ($100.0\%$)**
- **Maximum Active Weight Discrepancy**: **$0.000000000$**
- **Maximum Fused Risk Discrepancy**: **$0.000000000$**
- **Maximum DCRI Discrepancy**: **$0.000000000$**

---

## 3. Unavailable vs Low-Quality Boundary Distinction

| Property | Missing Modality ($A_i = 0$) | Degraded Quality Modality ($A_i = 1, Q_i = 0.0$) |
| :--- | :--- | :--- |
| **Availability Flag** | `availability = False` | `availability = True` |
| **Quality Score** | Contractually $Q_i = 0.0$ | $Q_i = 0.0$ |
| **Router Logit** | Masked to $\tilde{z}_i = -\infty$ | $z_i = \alpha C_i + \beta R_i - \gamma U_i + 0.0$ (finite) |
| **Authority Weight** | **$w_i = 0.000000$ strictly** | **$w_i > 0.000000$ (participates in softmax)** |
| **Clinical Semantics** | Channel does not exist for this case | Channel exists but image/signal is degraded |
