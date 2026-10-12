# Chapter 05: Cross-Modality Conflict Stratification

## 1. Conflict Metrics and Formal Definitions
Cross-modality prediction conflict occurs when constituent modality models emit divergent risk estimates for the same encounter. Dimension D formalizes continuous discordance and stratifies decision behavior across pre-defined disagreement tiers.

### Scope & Availability Regime
The conflict metrics in this chapter are evaluated exclusively across the complete tri-modal cohort ($\mathcal{A} = \{R, F, C\}$, Regime RFC, $N=500$, seed 115). Because pairwise divergence requires at least two active channels, pairwise discordance metrics are not defined for unimodal singleton or empty regimes.

For an encounter with active modality set $\mathcal{A} = \{R, F, C\}$, the discordance metrics are defined as:
1. **Maximum Pairwise Disagreement ($\Delta_{\max}$)**:
   $$\Delta_{\max} = \max_{i, j \in \mathcal{A}, i < j} |r_i - r_j|$$
2. **Mean Pairwise Disagreement ($\Delta_{\text{mean}}$)**:
   $$\Delta_{\text{mean}} = \frac{2}{|\mathcal{A}|(|\mathcal{A}| - 1)} \sum_{i < j \in \mathcal{A}} |r_i - r_j|$$
3. **Weighted Consensus Variance ($\sigma_w$)**:
   $$\sigma_w = \sqrt{\sum_{i \in \mathcal{A}} w_i (r_i - R_{\text{fusion}})^2}$$
4. **Conflict Severity Score (CSS)**:
   $$\text{CSS} \equiv \Delta_{\max}$$

### 1.1 Conflict Strata Definitions
Encounter packets are partitioned into three operational conflict strata based on CSS:
- **Low Conflict**: $\text{CSS} < 0.20$ (Substantial inter-modality agreement)
- **Moderate Conflict**: $0.20 \le \text{CSS} < 0.35$ (Partial inter-modality disagreement)
- **High Conflict**: $\text{CSS} \ge 0.35$ (High prediction disagreement)

The thresholds $0.20$ and $0.35$ represent frozen operational stratification boundaries declared in the evaluation protocol; they are not externally validated clinical diagnostic thresholds.

## 2. Quantitative Conflict Stratification (ACARA-U B6)
The distribution of conflict strata and corresponding decision behaviors across the $N=500$ tri-modal cohort is summarized below:

| Conflict Stratum | Packet Count | Cohort % | Mean CSS $\overline{\Delta}_{\max}$ | Mean Weighted Spread $\overline{\sigma}_w$ | Mean Risk $\overline{R}_{\text{fusion}}$ | Mean DCRI $\overline{\text{DCRI}}$ | Tier 0 (Routine) | Tier 1 (Assessment) | Tier 2 (Escalation) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LOW** | 45 | 9.0% | 0.141305 | 0.059312 | 0.118513 | 0.056217 | 43 (95.6%) | 2 (4.4%) | 0 (0.0%) |
| **MODERATE** | 92 | 18.4% | 0.288083 | 0.117979 | 0.187526 | 0.116403 | 74 (80.4%) | 17 (18.5%) | 1 (1.1%) |
| **HIGH** | 363 | 72.6% | 0.626179 | 0.259545 | 0.337092 | 0.275521 | 115 (31.7%) | 168 (46.3%) | 80 (22.0%) |
| **Total / Overall** | **500** | **100.0%** | **0.520286** | **0.215570** | **0.289900** | **0.226506** | **232 (46.4%)** | **187 (37.4%)** | **81 (16.2%)** |

## 3. Decision Mechanics Across Conflict Strata

1. **Low-Disagreement Cohort**: When modality predictions closely agree ($\text{CSS} < 0.20$, $N=45$), mean weighted spread $\sigma_w$ is low ($0.059312$), and $95.6\%$ of packets map algorithmically to Tier 0 (Routine Review) with zero Tier 2 escalations. This reflects consistent low-risk model agreement; it does not independently establish that routine review was clinically appropriate for all 43 cases.
2. **Moderate-Disagreement Cohort**: Under intermediate divergence ($0.20 \le \text{CSS} < 0.35$, $N=92$), spread expands to $0.117979$, Tier 1 Assessment increases to $18.5\%$, and 1 packet escalates to Tier 2.
3. **High-Disagreement Cohort**: Under wide prediction disparity ($\text{CSS} \ge 0.35$, $N=363$), weighted spread rises to $0.259545$, and higher operational tiers dominate ($46.3\%$ Tier 1, $22.0\%$ Tier 2 Escalation). In this stratum, ACARA-U routes greater weight toward channels with higher reliability priors and lower sample uncertainties. However, the fact that a modality receives greater routing weight reflects architectural design rules, not proof that the favored modality's prediction matches true patient condition.
