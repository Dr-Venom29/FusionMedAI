# Chapter 08: Availability Regime Breakdown

## 1. Multi-Regime Evaluation Design

In practical clinical deployments, patients present with heterogeneous subsets of modalities. To determine how component ablations behave across all missing-modality scenarios, every packet in the $N=500$ cohort was systematically evaluated across all $2^3 = 8$ potential availability regimes:

$$\mathcal{P}(\{\text{retina}, \text{foot}, \text{clinical}\}) = \{R, F, C, RF, RC, FC, RFC, \text{EMPTY}\}$$

This generates $500 \times 8 = 4,000$ evaluations per candidate variant.

---

## 2. Regime-Stratified Results Across Variants

The table below reports mean fused risk ($\overline{R}$), mean DCRI ($\overline{\text{DCRI}}$), and mean channel weights across all 8 availability regimes:

| Regime | Active Set $\mathcal{A}$ | Variant | Mean $w_{\text{retina}}$ | Mean $w_{\text{foot}}$ | Mean $w_{\text{clinical}}$ | Mean Fused Risk $\overline{R}$ | Mean DCRI |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **R** | Retina only | **A0 (Ref)** | $1.0000$ | $0.0000$ | $0.0000$ | $0.2550$ | $0.2549$ |
| | | **A1 ($\alpha=0$)** | $1.0000$ | $0.0000$ | $0.0000$ | $0.2550$ | $0.2549$ |
| | | **A3 ($\gamma=0$)** | $1.0000$ | $0.0000$ | $0.0000$ | $0.2550$ | $0.2549$ |
| | | **A5 ($\delta=0$)** | $1.0000$ | $0.0000$ | $0.0000$ | $0.2550$ | $0.2550$ |
| | | **A_UNIFORM** | $1.0000$ | $0.0000$ | $0.0000$ | $0.2550$ | $0.2549$ |
| **F** | Foot only | **A0 (Ref)** | $0.0000$ | $1.0000$ | $0.0000$ | $0.5219$ | $0.4625$ |
| | | **A1 ($\alpha=0$)** | $0.0000$ | $1.0000$ | $0.0000$ | $0.5219$ | $0.4625$ |
| | | **A3 ($\gamma=0$)** | $0.0000$ | $1.0000$ | $0.0000$ | $0.5219$ | $0.4625$ |
| | | **A5 ($\delta=0$)** | $0.0000$ | $1.0000$ | $0.0000$ | $0.5219$ | $0.5219$ |
| | | **A_UNIFORM** | $0.0000$ | $1.0000$ | $0.0000$ | $0.5219$ | $0.4625$ |
| **C** | Clinical only | **A0 (Ref)** | $0.0000$ | $0.0000$ | $1.0000$ | $0.1135$ | $0.1096$ |
| | | **A1 ($\alpha=0$)** | $0.0000$ | $0.0000$ | $1.0000$ | $0.1135$ | $0.1096$ |
| | | **A3 ($\gamma=0$)** | $0.0000$ | $0.0000$ | $1.0000$ | $0.1135$ | $0.1096$ |
| | | **A5 ($\delta=0$)** | $0.0000$ | $0.0000$ | $1.0000$ | $0.1135$ | $0.1135$ |
| | | **A_UNIFORM** | $0.0000$ | $0.0000$ | $1.0000$ | $0.1135$ | $0.1096$ |
| **RF** | Retina + Foot | **A0 (Ref)** | $0.6578$ | $0.3422$ | $0.0000$ | $0.3470$ | $0.2875$ |
| | | **A1 ($\alpha=0$)** | $0.6503$ | $0.3497$ | $0.0000$ | $0.3557$ | $0.2962$ |
| | | **A3 ($\gamma=0$)** | $0.5237$ | $0.4763$ | $0.0000$ | $0.3747$ | $0.3151$ |
| | | **A5 ($\delta=0$)** | $0.6578$ | $0.3422$ | $0.0000$ | $0.3470$ | $0.3470$ |
| | | **A_UNIFORM** | $0.5000$ | $0.5000$ | $0.0000$ | $0.3885$ | $0.3289$ |
| **RC** | Retina + Clinical | **A0 (Ref)** | $0.6754$ | $0.0000$ | $0.3246$ | $0.1998$ | $0.1958$ |
| | | **A1 ($\alpha=0$)** | $0.5607$ | $0.0000$ | $0.4393$ | $0.1926$ | $0.1886$ |
| | | **A3 ($\gamma=0$)** | $0.6673$ | $0.0000$ | $0.3327$ | $0.1987$ | $0.1947$ |
| | | **A5 ($\delta=0$)** | $0.6754$ | $0.0000$ | $0.3246$ | $0.1998$ | $0.1998$ |
| | | **A_UNIFORM** | $0.5000$ | $0.0000$ | $0.5000$ | $0.1843$ | $0.1803$ |
| **FC** | Foot + Clinical | **A0 (Ref)** | $0.0000$ | $0.5141$ | $0.4859$ | $0.3349$ | $0.2717$ |
| | | **A1 ($\alpha=0$)** | $0.0000$ | $0.4062$ | $0.5938$ | $0.2854$ | $0.2221$ |
| | | **A3 ($\gamma=0$)** | $0.0000$ | $0.6479$ | $0.3521$ | $0.3833$ | $0.3200$ |
| | | **A5 ($\delta=0$)** | $0.0000$ | $0.5141$ | $0.4859$ | $0.3349$ | $0.3349$ |
| | | **A_UNIFORM** | $0.0000$ | $0.5000$ | $0.5000$ | $0.3177$ | $0.2544$ |
| **RFC** | Tri-modal | **A0 (Ref)** | $0.5010$ | $0.2609$ | $0.2381$ | $0.2899$ | $0.2265$ |
| | | **A1 ($\alpha=0$)** | $0.4301$ | $0.2330$ | $0.3369$ | $0.2748$ | $0.2114$ |
| | | **A3 ($\gamma=0$)** | $0.4180$ | $0.3776$ | $0.2045$ | $0.3207$ | $0.2573$ |
| | | **A5 ($\delta=0$)** | $0.5010$ | $0.2609$ | $0.2381$ | $0.2899$ | $0.2899$ |
| | | **A_UNIFORM** | $0.3333$ | $0.3333$ | $0.3333$ | $0.2968$ | $0.2334$ |
| **EMPTY** | All Missing | **All Variants** | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ |

---

## 3. Mathematical Implications

### 3.1 Unimodal Singleton Invariance (Gate C16-05)
In any singleton regime $|\mathcal{A}| = 1$ (e.g., $R, F, C$):
$$\tilde{z}_i = z_i, \quad \tilde{z}_{j \neq i} = -\infty \implies w_i = \frac{\exp(z_i)}{\exp(z_i)} \equiv 1.000000$$

Consequently, regardless of the values of $\alpha, \beta, \gamma, \eta$, routing weights are mathematically locked to unity ($w_i \equiv 1.0$), and fused risk is invariant ($\Delta R \equiv 0.0$). Router coefficient ablations exert influence **strictly and exclusively** during multi-modality encounters ($|\mathcal{A}| \ge 2$).

### 3.2 Downstream Discount Behavior Across Modalities
In singleton regimes, the downstream discount shifts DCRI by exactly $+\delta^* U_i$:
- In **Regime R** ($U_{\text{retina}} \approx 0.001$): $\Delta \text{DCRI} = +0.0001$ ($0.2549 \to 0.2550$).
- In **Regime F** ($U_{\text{foot}} \approx 0.594$): $\Delta \text{DCRI} = +0.0594$ ($0.4625 \to 0.5219$).
- In **Regime C** ($U_{\text{clinical}} \approx 0.039$): $\Delta \text{DCRI} = +0.0039$ ($0.1096 \to 0.1135$).

This proves that the DCRI discount operates as designed: applying maximum conservatism to encounters dominated by high-uncertainty modalities (such as foot ulcers) while imposing negligible penalty on high-certainty modalities.

### 3.3 Fail-Closed EMPTY Invariance (Gate C16-11)
When no modalities are available ($\mathcal{A} = \emptyset$), every variant safely aborts to the fail-closed fallback:
$$w_i = 0.0 \quad \forall i, \quad R_{\text{fusion}} = 0.0, \quad \text{DCRI} = 0.0, \quad \text{Tier} = \text{Routine}$$
Zero exceptions or non-finite values occurred across all $500 \times 7 = 3,500$ empty evaluations.
