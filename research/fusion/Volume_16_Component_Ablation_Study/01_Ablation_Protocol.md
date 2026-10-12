# Chapter 01: ACARA-U Component Ablation Protocol

## 1. Executive Summary

Phase C11.16 executes the formal component ablation study of the **ACARA-U v2 Dynamic Multimodal Decision Router** and its downstream **Dynamic Composite Risk Index (DCRI)** discount mechanism.

The primary objective is to decompose the decision-level fusion architecture, quantify the exact marginal contribution of each routing and discounting term, and verify that the system preserves safety invariants across all perturbations.

In strict adherence to confirmatory research principles:
- No hyperparameters, coefficients, or decision thresholds are re-optimized or tuned.
- Deletions are conducted strictly one-at-a-time without renormalizing remaining coefficients.
- All evaluations are executed over the frozen $N=500$ decision packets upstream benchmark cohort (seed 115).
- A diagnostic baseline ($A_{\text{UNIFORM}}$) and a negative control fault injection ($A6$) are included to establish bounds and test verifier sensitivity.

---

## 2. Theoretical Mathematical Framework

### 2.1 Full Reference Architecture ($B6 / A0$)
The ACARA-U router maps multimodal encounter characteristics to decision weights $w = (w_{\text{retina}}, w_{\text{foot}}, w_{\text{clinical}}) \in \Delta^2$ on the 2-simplex:

$$\mathcal{A} \subseteq \{\text{retina}, \text{foot}, \text{clinical}\}, \quad |\mathcal{A}| \ge 1$$

For each channel $i \in \{\text{retina}, \text{foot}, \text{clinical}\}$, the raw logit is computed as:
$$z_i = \alpha C_i + \beta R_i - \gamma U_i + \eta Q_i$$

where:
- $C_i \in [0, 1]$ represents modality confidence (e.g., maximum softmax probability or calibrated decision confidence).
- $R_i \in [0, 1]$ represents global model reliability (empirically frozen priors: $R_{\text{retina}}=0.85, R_{\text{foot}}=0.80, R_{\text{clinical}}=0.75$).
- $U_i \in [0, 1]$ represents epistemic predictive uncertainty.
- $Q_i \in [0, 1]$ represents input signal quality.

### 2.2 Availability Masking & Softmax Normalization
Availability masking enforces that unavailable modalities cannot influence routing weights:

$$\tilde{z}_i = \begin{cases} z_i & \text{if } i \in \mathcal{A} \\ -\infty & \text{if } i \notin \mathcal{A} \end{cases}$$

The channel decision weights are obtained via softmax over the active set:

$$w_i = \begin{cases} \frac{\exp(z_i)}{\sum_{j \in \mathcal{A}} \exp(z_j)} & \text{if } i \in \mathcal{A} \\ 0.0 & \text{if } i \notin \mathcal{A} \end{cases}$$

For the degenerate case where no modalities are available ($\mathcal{A} = \emptyset$), the system fails closed:
$$w_i = 0.0 \quad \forall i, \quad R_{\text{fusion}} = 0.0, \quad \text{DCRI} = 0.0, \quad \text{Status} = \text{NO\_MODALITY\_AVAILABLE}$$

### 2.3 Fused Risk and Downstream DCRI Aggregation
The fused risk estimate is the linear combination of channel risk predictions $r_i \in [0, 1]$:
$$R_{\text{fusion}} = \sum_{i \in \mathcal{A}} w_i r_i$$

The epistemic uncertainty penalty is defined as:
$$P_U = \delta^* \sum_{i \in \mathcal{A}} U_i$$

where $\delta^* = 0.10$ is the prespecified downstream uncertainty discount parameter. Note that the canonical DCRI definition (Phase C11.6, C11.13, C11.15) aggregates unweighted channel uncertainties across available modalities $\mathcal{A}$, ensuring that the downstream uncertainty discount is invariant to routing weight allocations ($\Delta P_U = 0$ for all router ablations).

The Dynamic Composite Risk Index is then computed as:
$$\text{DCRI} = R_{\text{fusion}} - P_U = \sum_{i \in \mathcal{A}} w_i r_i - \delta^* \sum_{i \in \mathcal{A}} U_i$$

---

## 3. Pre-Specified Ablation Candidates

The ablation matrix evaluates 8 distinct configurations:

| Candidate ID | Target Component | Modification to Frozen Parameters | Scientific Question Addressed |
| :--- | :--- | :--- | :--- |
| **A0** | Reference Architecture | None ($\alpha=1.0, \beta=1.5, \gamma=1.0, \eta=0.5, \delta=0.10$) | Reproduces frozen Phase C11.13/C11.15 benchmark? |
| **A1** | Modality Confidence | $\alpha = 0.0$ ($\beta=1.5, \gamma=1.0, \eta=0.5, \delta=0.10$) | What is the marginal influence of dynamic confidence $C_i$? |
| **A2** | Reliability Prior | $\beta = 0.0$ ($\alpha=1.0, \gamma=1.0, \eta=0.5, \delta=0.10$) | What is the marginal influence of historical reliability prior $R_i$? |
| **A3** | Predictive Uncertainty | $\gamma = 0.0$ ($\alpha=1.0, \beta=1.5, \eta=0.5, \delta=0.10$) | What is the marginal effect of penalizing high-uncertainty models in the router? |
| **A4** | Input Signal Quality | $\eta = 0.0$ ($\alpha=1.0, \beta=1.5, \gamma=1.0, \delta=0.10$) | What is the marginal contribution of the input quality bonus $Q_i$? |
| **A5** | DCRI Uncertainty Discount | $\delta = 0.0$ ($\alpha=1.0, \beta=1.5, \gamma=1.0, \eta=0.5$) | How does the downstream discount affect score levels, tier shifts, and negative values? |
| **A_UNIFORM** | Diagnostic Normalization | $\alpha=\beta=\gamma=\eta=0.0, \delta=0.10$ | What occurs when all available modalities are weighted uniformly ($w_i = 1/|\mathcal{A}|$)? |
| **A6** | Negative Control Fault Injection | Deliberate availability breach ($w_{\text{inactive}} = 0.10$) | Does the acceptance gate reliably detect and reject illegal weight allocations? |

---

## 4. Evaluation Protocol and Datasets

### 4.1 Cohort Specification
- **Dataset:** Canonical controlled decision packet cohort generated under seed 115.
- **Sample Size:** $N = 500$ packets.
- **SHA-256 Checksum:** `1ff37fbeeba33feae7bf14d57d0df4a10df48ac789edeedf9dbd2fa8121382bb`
- **Reference File:** `experiments/fusion/baseline_comparison/packet_manifest.json`

### 4.2 Availability Regime Stratification
Each candidate variant is evaluated under two distinct protocols:
1. **Full-Cohort Reference Availability:** Uses the natural availability masks encoded in the canonical 500 packets ($430$ tri-modal, $70$ bi-modal/unimodal).
2. **Regime-Stratified Grid:** Each of the 500 packets is re-evaluated across all 8 possible availability regimes:
   - $R$: Retina only ($|\mathcal{A}|=1$)
   - $F$: Foot ulcer only ($|\mathcal{A}|=1$)
   - $C$: Clinical tabular only ($|\mathcal{A}|=1$)
   - $RF$: Retina + Foot ulcer ($|\mathcal{A}|=2$)
   - $RC$: Retina + Clinical tabular ($|\mathcal{A}|=2$)
   - $FC$: Foot ulcer + Clinical tabular ($|\mathcal{A}|=2$)
   - $RFC$: Full tri-modal encounter ($|\mathcal{A}|=3$)
   - $\text{EMPTY}$: All modalities missing ($\mathcal{A}=\emptyset$)

Total evaluations per variant: $500 \times 8 = 4,000$ decisions.

### 4.3 Hypothetical Decision Triage Tiers
To quantify operational triage impact without asserting clinical efficacy, decisions are mapped to 3 hypothetical risk tiers:
- **Tier 0 (Routine Review):** $\text{DCRI} < 0.20$ (Reference $A0$: $232$ packets, $46.4\%$)
- **Tier 1 (Additional Assessment):** $0.20 \le \text{DCRI} < 0.40$ (Reference $A0$: $187$ packets, $37.4\%$)
- **Tier 2 (Escalation for Human Review):** $\text{DCRI} \ge 0.40$ (Reference $A0$: $81$ packets, $16.2\%$)

Transition matrices ($3 \times 3$) track packet movement between $A0$ and each ablated variant $Av$.

---

## 5. Acceptance Gates (C16-01 through C16-12)

The study establishes 12 mandatory acceptance criteria:
1. **C16-01 (Manifest Integrity):** All 8 output artifacts exist and match SHA-256 manifest.
2. **C16-02 (Reference Reproduction):** $A0$ reproduces $B6$ within $10^{-6}$ error on $N=500$.
3. **C16-03 (Simplex Preservation):** Valid variants exhibit exactly 0 simplex and 0 masking violations.
4. **C16-04 (Negative Control Rejection):** $A6$ is 100% detected and rejected.
5. **C16-05 (Unimodal Invariance):** In singleton regimes, $\Delta w_i \equiv 0.0$ and $\Delta R \equiv 0.0$ for $A1$–$A4$.
6. **C16-06 (DCRI Orthogonality):** $A5$ produces $\Delta w \equiv 0.0$ and $\Delta R \equiv 0.0$, shifting DCRI by exactly $+\delta^* P_U$.
7. **C16-07 (Uncertainty Directionality):** Removing $\gamma$ ($A3$) increases foot ulcer weight and fused risk.
8. **C16-08 (Attribution Hierarchy):** Weight shift magnitudes obey: $\text{Uncertainty}, \text{Confidence} > \text{Reliability} > \text{Quality}$.
9. **C16-09 (Decision Tier Stability):** Transition matrices sum to 500 with zero paradoxical 2-tier jumps.
10. **C16-10 (Bootstrap Determinism):** Paired bootstrap ($B=1000$, seed 7116) produces valid coverage.
11. **C16-11 (Fail-Closed Invariance):** $\text{EMPTY}$ regime yields $w=0, R=0, \text{DCRI}=0$ across all variants.
12. **C16-12 (Documentation Completeness):** All 13 chapters and README are fully authored.
