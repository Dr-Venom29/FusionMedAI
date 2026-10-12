# Chapter 07: Epistemic Uncertainty and DCRI Behavior

## 1. Mathematical Formulation of DCRI
The Dual-Constraint Risk Index (DCRI) is defined as an algorithmic decision score intended to discount predicted disease risk by an estimate of epistemic uncertainty.

In Phase C11.6 and reaffirmed in Phase C11.13, the unweighted linear uncertainty penalty was selected for evaluation:
$$P_U = \delta^* \sum_{i \in \mathcal{A}} U_i, \quad \delta^* = 0.10$$
$$\text{DCRI} = R_{\text{fusion}} - P_U = \left( \sum_{i \in \mathcal{A}} w_i r_i \right) - 0.10 \left( \sum_{i \in \mathcal{A}} U_i \right)$$

### 1.1 Unweighted Formulation Properties & Trade-Offs
1. **Unweighted Sum**: The term "unweighted" signifies that $P_U$ sums raw channel uncertainties $U_i$ across active modalities without multiplying each uncertainty by its routing weight $w_i$. This is a specific research formulation chosen for evaluation; it is not a universally accepted clinical or statistical standard.
2. **Channel Weighting Independence**: Because $P_U$ does not scale with $w_i$, the penalty value is insensitive to routing weight redistributions among the available modalities.
3. **Missing-Modality Implication**: When a modality is unavailable ($i \notin \mathcal{A}$), its uncertainty $U_i$ is omitted from the sum. This reduces the total penalty when diagnostic evidence is missing. While this avoids double-penalizing missing modalities alongside routing adjustments, it also means encounters with missing channels receive smaller penalties than complete tri-modal encounters with identical available channel uncertainties. This sensitivity is an architectural characteristic that requires careful clinical consideration.

## 2. Quantitative Characteristics of the Uncertainty Penalty
Across the $N=500$ evaluation cohort (seed 115) under full tri-modal availability:
- **Mean Uncertainty Sum $\sum_{i \in \mathcal{A}} U_i$**: $0.633936$
- **Mean Penalty $\overline{P}_U$**: $0.063394$
- **Minimum Observed Penalty**: $0.007473$
- **Maximum Observed Penalty**: $0.102808$
- **Standard Deviation of Penalty**: $0.024585$

## 3. Analysis of Negative DCRI Scores

### 3.1 Prevalence and Mathematical Mechanics
Because $\text{DCRI} = R_{\text{fusion}} - P_U$, any encounter where the uncertainty penalty exceeds the predicted fused risk ($P_U > R_{\text{fusion}}$) produces a negative DCRI value ($\text{DCRI} < 0$).

In the reference ACARA-U (B6) cohort:
- Exactly **37 out of 500 packets (7.4%)** generate negative DCRI values.
- **Minimum Observed DCRI**: $-0.027143$.
- **Mean of Negative Scores**: $-0.010750$.
- **Maximum Negative Score**: $-0.000041$.

The general mathematical condition is simply $P_U > R_{\text{fusion}}$. On this cohort, this occurs predominantly when baseline fused risk is low ($R_{\text{fusion}} < 0.05$) while active modalities output moderate-to-high model uncertainties ($\sum U_i > 0.50$).

### 3.2 Non-Clamping Policy
In accordance with the frozen evaluation protocol, negative DCRI scores are preserved unclamped in all raw metrics, tables, and serialized records. Post-hoc clamping would conceal the presence of high epistemic uncertainty in low-risk encounters.

### 3.3 Decision Tier Assignment and Future Work
All 37 negative DCRI scores fall into **Tier 0: Routine Review** under the frozen operational threshold definition ($\text{DCRI} < \tau_1 = 0.20$). 

However, falling into Tier 0 is a consequence of threshold mapping and does not by itself establish clinical safety: subtracting uncertainty to push an encounter into routine review is a counter-intuitive behavior in clinical risk screening. Prospective proposals (such as a rectified zero-floor $\text{DCRI}_{\text{rectified}} = \max(0.0, \text{DCRI})$ or separate two-dimensional uncertainty routing) are designated for subsequent investigation in Phase C11.18. Such transformations will not be applied retrospectively to the frozen C11.17 results, and any future display modifications must preserve raw scores in audit logs to ensure auditability.
