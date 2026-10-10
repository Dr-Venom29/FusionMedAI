# Combination-Level Routing & Information Metrics

## 1. Decision Authority Allocation by Combination

Under ACARA-U, routing weights $w_i$ are assigned dynamically via softmax over active channel scores:

$$
z_i = \alpha C_i + \beta R_i - \gamma U_i + \eta Q_i, \quad w_i = \frac{\exp(z_i)}{\sum_{j \in \mathcal{A}} \exp(z_j)}
$$

Unavailable channels receive hard masking ($A_i = 0 \implies \tilde{z}_i = -\infty \implies w_i = 0.0$).

### Empirical Mean Weight Allocation (Cohort Evaluation across D2)

| Combination | Modality Set | Mean $w_{\text{retina}}$ | Mean $w_{\text{foot}}$ | Mean $w_{\text{clinical}}$ | $\sum w_i$ | Mean Routing Entropy $H(w)$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RFC** | Tri-modal | $0.4357$ | $0.3478$ | $0.2165$ | **1.0000** | **0.9996** |
| **RF** | Bimodal | $0.5562$ | $0.4438$ | $0.0000$ | **1.0000** | **0.6160** |
| **RC** | Bimodal | $0.6698$ | $0.0000$ | $0.3302$ | **1.0000** | **0.6558** |
| **FC** | Bimodal | $0.0000$ | $0.6276$ | $0.3724$ | **1.0000** | **0.6401** |
| **R** | Unimodal | $1.0000$ | $0.0000$ | $0.0000$ | **1.0000** | **0.0000** |
| **F** | Unimodal | $0.0000$ | $1.0000$ | $0.0000$ | **1.0000** | **0.0000** |
| **C** | Unimodal | $0.0000$ | $0.0000$ | $1.0000$ | **1.0000** | **0.0000** |

---

## 2. Shannon Routing Entropy Analysis

The Shannon routing entropy measures authority dispersion across active modalities:

$$
H(w) = -\sum_{i \in \mathcal{M}, w_i > 0} w_i \ln(w_i)
$$

### Mathematical Invariants & Empirical Findings:
1. **Tri-modal Maximum**: Under RFC, authority is distributed across all 3 channels ($\overline{H} = 0.9996 \pm 0.0384$).
2. **Bimodal Boundedness**: Under RF, RC, and FC, authority is distributed across 2 channels with entropy bounded by $\ln(2) \approx 0.6931$ ($\overline{H} \in [0.6160, 0.6558]$).
3. **Unimodal Zero Invariant**: Under R, F, and C, authority is concentrated into a single channel ($w_i = 1.0 \implies H = -1.0 \ln(1.0) = 0.0000 \pm 0.0000$).

This confirms that the routing layer naturally concentrates authority as cardinality decreases, reducing to deterministic single-modality authority for unimodal inputs.

> **Entropy Interpretation (Conditional vs Global Mixture)**:  
> Routing entropy is zero for a unimodal packet because its sole available modality receives all routing authority ($w_i = 1.0 \implies H=0$). The global mean entropy across the entire population, however, depends on the frequency-weighted mixture of modality combinations ($\overline{H} = 0.4154$ in D1, $0.6662$ in D2, $0.7737$ in D3). Its increase across D1–D3 reflects the changing composition of the evaluated distributions (e.g. higher proportions of bimodal and complete combinations relative to uniform allocation) and must not be interpreted as an increase in entropy within unimodal tail packets. RQ3 is evaluated through combination- or tier-conditional comparisons; global means provide complementary distribution-level summaries.
