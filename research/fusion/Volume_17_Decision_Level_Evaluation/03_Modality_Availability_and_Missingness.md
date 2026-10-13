# Chapter 03: Modality Availability and Missingness

## 1. Availability Regimes Architecture
In clinical practice, sensor failure, patient non-compliance, or clinical omission often lead to missing diagnostic modalities. Dimension B evaluates ACARA-U (B6) across all eight possible modality availability regimes:
- **Tri-modal**: $\mathcal{A} = \{R, F, C\}$ (Regime RFC)
- **Dual-modal**: $\{R, F\}, \{R, C\}, \{F, C\}$ (Regimes RF, RC, FC)
- **Unimodal Singletons**: $\{R\}, \{F\}, \{C\}$ (Regimes R, F, C)
- **Empty Input**: $\mathcal{A} = \emptyset$ (Regime EMPTY)

### Experimental Design (Paired Evaluation)
The evaluation evaluates the exact same 500 packet identities ($N=500$, seed 115) across all eight regimes. For each packet, only the availability mask is altered; the underlying modality signals, confidences, uncertainties, and qualities remain identical. Comparisons across regimes are therefore paired at the packet level.

## 2. Quantitative Evaluation Across Availability Regimes (B6)
Evaluating the paired $N=500$ cohort across all eight regimes yields:

| Regime | Active Set $\mathcal{A}$ | $\lvert \mathcal{A} \rvert$ | Mean Risk $\overline{R}_{\mathrm{fusion}}$ | Mean Penalty $\overline{P}_U$ | Mean DCRI $\overline{\mathrm{DCRI}}$ | Negative DCRI Count | Inactive Weight Leakage Check |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RFC** | $\{R, F, C\}$ | 3 | 0.289900 | 0.063394 | 0.226506 | 37 (7.4%) | 0 / 500 violations ($w_m \le 10^{-7}$) |
| **RF** | $\{R, F\}$ | 2 | 0.297424 | 0.044769 | 0.252655 | 18 (3.6%) | 0 / 500 violations ($w_C \le 10^{-7}$) |
| **RC** | $\{R, C\}$ | 2 | 0.286940 | 0.040524 | 0.246416 | 13 (2.6%) | 0 / 500 violations ($w_F \le 10^{-7}$) |
| **FC** | $\{F, C\}$ | 2 | 0.316447 | 0.040239 | 0.276208 | 7 (1.4%) | 0 / 500 violations ($w_R \le 10^{-7}$) |
| **R** | $\{R\}$ | 1 | 0.255037 | 0.019089 | 0.235948 | 15 (3.0%) | 0 / 500 violations ($w_{F,C} \le 10^{-7}$) |
| **F** | $\{F\}$ | 1 | 0.334005 | 0.021848 | 0.312157 | 6 (1.2%) | 0 / 500 violations ($w_{R,C} \le 10^{-7}$) |
| **C** | $\{C\}$ | 1 | 0.301369 | 0.017127 | 0.284242 | 4 (0.8%) | 0 / 500 violations ($w_{R,F} \le 10^{-7}$) |
| **EMPTY** | $\emptyset$ | 0 | 0.000000 | 0.000000 | 0.000000 | 0 (0.0%) | 0 / 500 violations ($w_{all} = 0.0$) |

*Note on Leakage Check*: The "Inactive Weight Leakage Check" verifies that for every packet in the regime, every inactive modality receives routing weight identically $\le 10^{-7}$ and is excluded from the fused risk inner product and uncertainty sum.

## 3. Mathematical Implications & Structural Limitations

### 3.1 Unimodal Singleton Invariance
When only a single modality channel is available ($\lvert\mathcal{A}\rvert = 1$), the active logit is $z_i$, while all inactive channels are hard-masked with $-\infty$. Under the masked softmax function:

$$
w_i = \frac{\exp(z_i)}{\exp(z_i)} \equiv 1.000000, \quad \forall j \notin \mathcal{A}: w_j \equiv 0.000000
$$

Under this formulation, the router weight for the available channel collapses mathematically to unity regardless of the dynamic coefficients $(\alpha, \beta, \gamma, \eta)$. Fused risk equals the single active channel's risk ($R_{\text{fusion}} = r_i$), and the uncertainty penalty equals that channel's discounted uncertainty ($P_U = \delta^* U_i$). The resulting score is therefore entirely dependent on the single available sensor's prediction and calibration.

### 3.2 Missingness vs Epistemic Uncertainty
Under the frozen unweighted penalty formulation ($P_U = \delta^* \sum_{i \in \mathcal{A}} U_i$), uncertainty is summed **only over available modalities**. Consequently:
- As modalities become unavailable, the mean penalty $\overline{P}_U$ decreases (from $0.063394$ in tri-modal RFC to $\sim 0.019$ in singletons).
- Missing modalities contribute zero uncertainty penalty. This design choice means missing evidence is not penalized as epistemic uncertainty; omitting a channel decreases the penalty rather than increasing it. Missingness reflects absent data, not an assessment that the missing channel was safe or reliable.

### 3.3 Fail-Closed EMPTY Software Contract
When no modalities are available ($\mathcal{A} = \emptyset$), standard softmax division is undefined. The engine enforces a defined software fallback:

$$
w_i = 0.0 \quad \forall i \in \{R, F, C\}, \quad R_{\text{fusion}} = 0.0, \quad P_U = 0.0, \quad \text{DCRI} = 0.0, \quad \text{Status} = \mathtt{NO\_MODALITY\_AVAILABLE}
$$

This behavior verifies that the software pipeline avoids NaN generation or unhandled division exceptions. In operational deployment, downstream systems must ensure that encounters flagged with `NO_MODALITY_AVAILABLE` trigger manual clinical triage rather than being treated as verified routine cases.
