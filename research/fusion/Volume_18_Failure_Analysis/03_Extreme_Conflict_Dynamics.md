# Chapter 03: Extreme Cross-Modality Conflict Dynamics

## 1. Conflict Stratification & Cohort Identification
Cross-modality disagreement is quantified via the maximum pairwise difference:
$$\Delta_{\max} = \max(|r_R - r_F|, |r_R - r_C|, |r_F - r_C|)$$
The 95th percentile cutoff across the $N=500$ cohort is $\Delta_{\max} = 0.942667$. The top 5% extreme-conflict cohort consists of exactly $N = 25$ packets, spanning from $0.9427$ to $0.9842$.

## 2. Triage Disposition under Extreme Disagreement
Under extreme discordance, ACARA-U routes encounters as follows:
- **Tier 2 (Escalation for Human Review)**: 7 packets (28.0%).
- **Tier 1 (Additional Assessment)**: 18 packets (72.0%).
- **Tier 0 (Routine Review)**: **0 packets (0.0%)**.

### Empirical Disposition on Evaluated Cohort
All 25 extreme-conflict packets ($\Delta_{\max} \ge 0.9427$) in the evaluated $N=500$ cohort were assigned to either Tier 1 or Tier 2. This supports the empirical finding that under the evaluated configuration, none of these 25 packets was assigned to Tier 0 (Routine Review).
- **Scope Boundary**: While this demonstrates that contradictory evidence was actively triaged to higher-acuity paths in this benchmark, it does not establish that contradictory clinical evidence can *never* be ignored under all operating conditions or future patient populations. Establishing such universal guarantees requires prospective multi-center clinical validation.

## 3. Modality Weight Allocation & Entropy
Under extreme conflict:
- Mean routing entropy is $1.042 \pm 0.038$ nats (indicating active multi-channel engagement).
- Mean modality weights:
  - $\bar{w}_{\mathrm{retina}} = 0.482 \pm 0.054$
  - $\bar{w}_{\mathrm{foot}} = 0.281 \pm 0.048$
  - $\bar{w}_{\mathrm{clinical}} = 0.237 \pm 0.039$
In this evaluated cohort, the router distributed authority across the three modalities rather than assigning all weight to a single modality. The appropriateness of these weights for clinical decision-making remains to be validated against suitable external data and patient outcomes.

## 4. Conflict Metric Provenance & Standardization: `css`, $\Delta_{\max}$, and $\Delta_{\mathrm{mean}}$
In the frozen baseline artifact `03_packet_level_results.csv`, the column labeled `css` was stored with values identical to `delta_max` ($\Delta_{\max}$):
- **`css`**: Legacy stored field name from earlier experimental versions; empirically identical to `delta_max` in the inspected artifact ($\max |\text{css} - \Delta_{\max}| = 0.0$). It is not an independent measurement.
- **$\Delta_{\max}$ (Maximum Pairwise Disparity)**: Defined as $\Delta_{\max} = \max(|r_R - r_F|, |r_R - r_C|, |r_F - r_C|)$.
- **$\Delta_{\mathrm{mean}}$ (Mean Pairwise Disparity)**: The analytical mean pairwise distance across all three channel pairs:
  $$\Delta_{\mathrm{mean}} = \frac{1}{3}(|r_R - r_F| + |r_R - r_C| + |r_F - r_C|)$$
  Comparing $\Delta_{\max}$ to $\Delta_{\mathrm{mean}}$ across the $N=500$ cohort:
  - Maximum difference: $\max |\Delta_{\max} - \Delta_{\mathrm{mean}}| = 0.330633$.
  - Mean difference: $0.158302$.
  - Spearman rank correlation: $\rho = 0.9412$ ($p < 10^{-100}$).

Both definitions and their provenance mapping are preserved across Volume 18 documentation.
