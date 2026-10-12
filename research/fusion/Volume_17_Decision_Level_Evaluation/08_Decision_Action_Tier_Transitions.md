# Chapter 08: Decision Action Tier Transitions

## 1. Decision Tier Transition Framework
The operational impact of an algorithmic fusion policy is expressed through the hypothetical action tiers assigned to encounters under pre-defined decision thresholds:
- **Tier 0 (Routine Review)**: $\text{DCRI} < 0.20$
- **Tier 1 (Additional Assessment)**: $0.20 \le \text{DCRI} < 0.40$
- **Tier 2 (Escalation for Human Review)**: $\text{DCRI} \ge 0.40$

This chapter examines the $3 \times 3$ transition matrices comparing the reference ACARA-U policy (B6) against each defined comparator baseline (B1–B5) across the canonical $N=500$ evaluation cohort (seed 115).

In each matrix:
- **Rows**: Reference Policy (B6) assigned tier
- **Columns**: Comparator Baseline ($B_k$) assigned tier
- **Diagonal Elements**: Packets with identical tier assignments under both policies
- **Off-Diagonal Elements (Above Diagonal)**: Comparator assigns a higher tier than B6 (comparator upgrade)
- **Off-Diagonal Elements (Below Diagonal)**: Comparator assigns a lower tier than B6 (comparator downgrade)

## 2. Transition Matrices Across Baseline Comparators

### 2.1 B6 vs B1 (Reliability-Selected Unimodal)
| B6 Tier (Row) \ B1 Tier (Col) | Tier 0 (Routine) | Tier 1 (Assessment) | Tier 2 (Escalation) | Total B6 |
| :--- | :---: | :---: | :---: | :---: |
| **Tier 0 (Routine)** | 209 | 23 | 0 | 232 |
| **Tier 1 (Assessment)** | 55 | 49 | 83 | 187 |
| **Tier 2 (Escalation)** | 11 | 12 | 58 | 81 |
| **Total B1** | **275** | **84** | **141** | **500** |

- **Unchanged**: 316 (63.2%)
- **Reclassified**: 184 (36.8%)
- **Comparator Upgrades (B1 > B6)**: 106 (21.2%)
- **Comparator Downgrades (B1 < B6)**: 78 (15.6%)
- **Two-Tier Transitions**: Exactly 11 packets jump across two tiers, all transitioning from B6 Tier 2 down to B1 Tier 0 (with 0 cases moving from B6 Tier 0 to B1 Tier 2). This reflects B1 winner-take-all routing collapsing to a single modality (Retina), ignoring complementary signals from other active channels.

### 2.2 B6 vs B2 (Uniform Average)
| B6 Tier (Row) \ B2 Tier (Col) | Tier 0 (Routine) | Tier 1 (Assessment) | Tier 2 (Escalation) | Total B6 |
| :--- | :---: | :---: | :---: | :---: |
| **Tier 0 (Routine)** | 194 | 38 | 0 | 232 |
| **Tier 1 (Assessment)** | 14 | 173 | 0 | 187 |
| **Tier 2 (Escalation)** | 0 | 29 | 52 | 81 |
| **Total B2** | **208** | **240** | **52** | **500** |

- **Unchanged**: 419 (83.8%)
- **Reclassified**: 81 (16.2%)
- **Comparator Upgrades (B2 > B6)**: 38 (7.6%)
- **Comparator Downgrades (B2 < B6)**: 43 (8.6%)
- **Two-Tier Transitions**: 0 (0.0%)
- *Note*: Uniform averaging concentrates assignments into Tier 1 (240 vs 187), assigning 29 fewer Tier 2 escalations than B6 (29 cases assigned to Tier 2 by B6 are assigned to Tier 1 by B2). Without external clinical gold-standard outcomes, this difference establishes policy divergence rather than proven clinical under-escalation.

### 2.3 B6 vs B3 (Confidence Fusion)
| B6 Tier (Row) \ B3 Tier (Col) | Tier 0 (Routine) | Tier 1 (Assessment) | Tier 2 (Escalation) | Total B6 |
| :--- | :---: | :---: | :---: | :---: |
| **Tier 0 (Routine)** | 195 | 37 | 0 | 232 |
| **Tier 1 (Assessment)** | 7 | 169 | 11 | 187 |
| **Tier 2 (Escalation)** | 0 | 2 | 79 | 81 |
| **Total B3** | **202** | **208** | **90** | **500** |

- **Unchanged**: 443 (88.6%)
- **Reclassified**: 57 (11.4%)
- **Comparator Upgrades (B3 > B6)**: 48 (9.6%)
- **Comparator Downgrades (B3 < B6)**: 9 (1.8%)
- **Two-Tier Transitions**: 0 (0.0%)

### 2.4 B6 vs B4 (Confidence + Reliability)
| B6 Tier (Row) \ B4 Tier (Col) | Tier 0 (Routine) | Tier 1 (Assessment) | Tier 2 (Escalation) | Total B6 |
| :--- | :---: | :---: | :---: | :---: |
| **Tier 0 (Routine)** | 192 | 40 | 0 | 232 |
| **Tier 1 (Assessment)** | 6 | 165 | 16 | 187 |
| **Tier 2 (Escalation)** | 0 | 2 | 79 | 81 |
| **Total B4** | **198** | **207** | **95** | **500** |

- **Unchanged**: 436 (87.2%)
- **Reclassified**: 64 (12.8%)
- **Comparator Upgrades (B4 > B6)**: 56 (11.2%)
- **Comparator Downgrades (B4 < B6)**: 8 (1.6%)
- **Two-Tier Transitions**: 0 (0.0%)

### 2.5 B6 vs B5 (Confidence + Reliability - Uncertainty)
| B6 Tier (Row) \ B5 Tier (Col) | Tier 0 (Routine) | Tier 1 (Assessment) | Tier 2 (Escalation) | Total B6 |
| :--- | :---: | :---: | :---: | :---: |
| **Tier 0 (Routine)** | 232 | 0 | 0 | 232 |
| **Tier 1 (Assessment)** | 2 | 185 | 0 | 187 |
| **Tier 2 (Escalation)** | 0 | 1 | 80 | 81 |
| **Total B5** | **234** | **186** | **80** | **500** |

- **Unchanged**: 497 (99.4%)
- **Reclassified**: 3 (0.6%)
- **Comparator Upgrades (B5 > B6)**: 0 (0.0%)
- **Comparator Downgrades (B5 < B6)**: 3 (0.6%)
- **Two-Tier Transitions**: 0 (0.0%)
- *Note*: Exactly 3 borderline packets shift tiers due to the inclusion of the adaptive quality bonus weight ($\eta = 0.5$ in B6 vs $\eta = 0.0$ in B5). This confirms a small observable effect on tier assignments under identical inputs; it does not by itself establish improved clinical efficacy.

## 3. Comparative Transition Summary
The table below synthesizes the transition dynamics observed between reference B6 and each comparator:

| Comparator Policy | Unchanged Count (%) | Reclassified Count (%) | Comparator Upgrades (%) | Comparator Downgrades (%) | Two-Tier Jumps ($0 \leftrightarrow 2$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **B1 (Unimodal WTA)** | 316 (63.2%) | 184 (36.8%) | 106 (21.2%) | 78 (15.6%) | 11 (B6 T2 $\to$ B1 T0) |
| **B2 (Uniform Avg)** | 419 (83.8%) | 81 (16.2%) | 38 (7.6%) | 43 (8.6%) | 0 (0.0%) |
| **B3 (Confidence)** | 443 (88.6%) | 57 (11.4%) | 48 (9.6%) | 9 (1.8%) | 0 (0.0%) |
| **B4 (Conf + Rel)** | 436 (87.2%) | 64 (12.8%) | 56 (11.2%) | 8 (1.6%) | 0 (0.0%) |
| **B5 (Conf + Rel - Unc)** | 497 (99.4%) | 3 (0.6%) | 0 (0.0%) | 3 (0.6%) | 0 (0.0%) |

### Observations:
1. **Mathematical Conservation**: All transition matrices conserve the full cohort size ($N=500$) and match both the B6 marginal distribution (Routine: 232, Assessment: 187, Escalation: 81) and each comparator's marginal distribution exactly.
2. **Observed Reclassification Ordering**: Total reclassifications relative to B6 follow the observed empirical ordering:
   $$\text{B1 } (36.8\%) > \text{B2 } (16.2\%) > \text{B4 } (12.8\%) > \text{B3 } (11.4\%) > \text{B5 } (0.6\%)$$
   Note that B4 has a slightly higher reclassification rate than B3, showing that adding reliability priors in B4 causes a larger shift from B6's balanced multi-attribute routing than confidence weighting alone.
3. **Limitation & Policy Interpretation**: Reclassification counts measure how differently policies assign the fixed cohort across decision boundaries. Because this evaluation lacks external clinical ground-truth diagnostic labels, a policy assigning more or fewer escalations cannot be declared clinically superior or safer based solely on transition counts.
