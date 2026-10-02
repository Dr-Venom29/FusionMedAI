# 04 ACARA-U Router Behavioral Evaluation & Benchmark — Phase C11.4

## 0. Evaluation Configuration

All numerical behavioral scenarios and operational matrix values in this document were generated using the standard full baseline configuration (B6):

$$\Theta = (\alpha, \beta, \gamma, \eta) = (1.0, 1.0, 1.0, 1.0)$$

- **Router Version**: `acarau_v2.0`
- **Frozen Validation Reliability Priors**:
  $$R_R = 0.929956, \quad R_F = 0.922266, \quad R_C = 0.825382$$

The reported weights are deterministic outputs of this exact configuration. No clinical labels, synthetic clinical targets, or optimization objectives are used to obtain the reported scenario values.

---

## 1. Controlled Behavioral Scenarios

To verify router mechanics without circular synthetic clinical targets, the router was benchmarked across four controlled stress scenarios:

### Scenario A: High Retina Trustworthiness
- **Input State**: Retina $C_R=0.95, U_R=0.05, Q_R=0.95$; Foot $C_F=0.60, U_F=0.40, Q_F=0.60$; Clinical $C_C=0.50, U_C=0.50, Q_C=0.50$.
- **Behavior**: $w_R = 0.633, w_F = 0.220, w_C = 0.148$. Retina captures dominant authority ($63.3\%$).

### Scenario B: Retina Predictive Uncertainty Increase
- **Perturbation**: $U_R$ elevated from $0.05 \to 0.80$ while all other signals remain fixed.
- **Behavior**: $w_R$ decreases from $0.633 \to 0.449$ (a $-29.1\%$ relative drop), dynamically shedding authority to Foot ($33.0\%$) and Clinical ($22.2\%$).

### Scenario C: Retina Modality Dropout ($A_R = 1 \to 0$)
- **Perturbation**: $A_R = 0, Q_R = 0.0$.
- **Behavior**: $w_R = 0.000$ strictly. Authority reallocated to Foot ($w_F = 0.598$) and Clinical ($w_C = 0.402$), with $\sum w_i = 1.000$.

### Scenario D: Clinical Tabular Missingness & Degradation
- **Perturbation**: Clinical feature quality degrades ($Q_C = 0.80 \to 0.20$).
- **Behavior**: $w_C$ drops from $0.298 \to 0.199$, automatically suppressing reliance on incomplete tabular records.

---

## 2. Seven Operational Modality Configurations Matrix

| Config ID | Description | Active Subset ($\mathcal{A}$) | $w_R$ | $w_F$ | $w_C$ | Sum | Routing Entropy $H(w)$ | Dominant Modality |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Config 1** | Full Tri-modal | $\{R, F, C\}$ | $0.397$ | $0.339$ | $0.265$ | $1.000$ | $1.085$ | Retina |
| **Config 2** | Imaging Only | $\{R, F\}$ | $0.539$ | $0.461$ | $0.000$ | $1.000$ | $0.690$ | Retina |
| **Config 3** | Retina + Clinical | $\{R, C\}$ | $0.600$ | $0.000$ | $0.400$ | $1.000$ | $0.673$ | Retina |
| **Config 4** | Foot + Clinical | $\{F, C\}$ | $0.000$ | $0.561$ | $0.439$ | $1.000$ | $0.686$ | Foot |
| **Config 5** | Retina Unimodal | $\{R\}$ | $1.000$ | $0.000$ | $0.000$ | $1.000$ | $0.000$ | Retina |
| **Config 6** | Foot Unimodal | $\{F\}$ | $0.000$ | $1.000$ | $0.000$ | $1.000$ | $0.000$ | Foot |
| **Config 7** | Clinical Unimodal | $\{C\}$ | $0.000$ | $0.000$ | $1.000$ | $1.000$ | $0.000$ | Clinical |
| **Config 0** | Zero Modality | $\{\emptyset\}$ | $0.000$ | $0.000$ | $0.000$ | $0.000$ | $0.000$ | None (`SAFE_REJECTION`) |

---

## 3. Monotonicity Grid Audits

- **Confidence Monotonicity**: Evaluated across finite step increments $C \in [0.1, 0.5, 0.9] \implies w_R \in [0.247, 0.368, 0.505]$ strictly ascending ($C_i \uparrow \implies w_i \uparrow$).
- **Reliability Ordering**: Identical inputs order weights as $w_R (0.354) > w_F (0.348) > w_C (0.297)$ matching frozen historical priors $R_R > R_F > R_C$.
- **Uncertainty Penalty Monotonicity**: Evaluated across finite step increments $U \in [0.1, 0.5, 0.9] \implies w_R \in [0.443, 0.315, 0.203]$ strictly descending ($U_i \uparrow \implies w_i \downarrow$).
- **Quality Monotonicity**: Evaluated across finite step increments $Q \in [0.1, 0.5, 0.9] \implies w_R \in [0.203, 0.315, 0.443]$ strictly ascending ($Q_i \uparrow \implies w_i \uparrow$).
