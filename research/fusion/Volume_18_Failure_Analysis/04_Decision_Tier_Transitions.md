# Chapter 04: Decision Tier Transitions & Comparative Reclassifications

## 1. Complete Paired Transition Matrix
Comparing Full ACARA-U (B6) against each baseline across the $N=500$ cohort:

| Comparator | Same Tier | B6 Upward (+1) | B6 Upward (+2) | B6 Downward (-1) | B6 Downward (-2) | Total Reclassified |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **B1 (Unimodal WTA)** | 316 (63.2%) | 67 (13.4%) | **11 (2.2%)** | 106 (21.2%) | **0 (0.0%)** | 184 (36.8%) |
| **B2 (Uniform Avg)** | 419 (83.8%) | 43 (8.6%) | 0 (0.0%) | 38 (7.6%) | 0 (0.0%) | 81 (16.2%) |
| **B3 (Confidence)** | 443 (88.6%) | 9 (1.8%) | 0 (0.0%) | 48 (9.6%) | 0 (0.0%) | 57 (11.4%) |
| **B4 (Conf + Rel)** | 436 (87.2%) | 8 (1.6%) | 0 (0.0%) | 56 (11.2%) | 0 (0.0%) | 64 (12.8%) |
| **B5 (Conf+Rel-Unc)**| 497 (99.4%) | 3 (0.6%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 3 (0.6%) |

*Note on Directionality*: Positive shift means the B6 tier is higher than the comparator's tier; negative shift means the B6 tier is lower ($\Delta_{\mathrm{Tier}} = \mathrm{Tier}_{B6} - \mathrm{Tier}_{B_k}$). These represent relative policy changes across $N=500$ controlled decision packets, not ground-truth clinical error labels.

## 2. Analysis of Two-Tier Jumps
- **B6 vs B1 Policy Discrepancy**: Exactly 11 packets shifted by two tiers upward from B1 to B6 (Tier 0 under B1 $\to$ Tier 2 under B6), with 0 downward two-tier shifts. In the packet-level records, these 11 encounters feature low retinal risk predictions ($r_R \le 0.12$, triggering Tier 0 under unimodal B1) alongside elevated complication risk in secondary modalities ($r_F, r_C \ge 0.75$). ACARA-U incorporates this multi-channel evidence, shifting the triage tier upward to Tier 2. This represents a substantial policy-dependent reclassification pattern rather than a clinically confirmed error label.
- **Absence of Two-Tier Jumps against Multi-Channel Baselines**: Against all multi-channel fusion strategies (B2, B3, B4, B5), ACARA-U produces **zero two-tier jumps** in either direction ($0 / 500$, $0.0\%$). In the evaluated 500-packet cohort, transitions between ACARA-U and other multi-channel approaches are confined to single-step boundary adjustments.

## 3. Directional Asymmetry Analysis
- **Observed Shifts vs B3 and B4**: Relative to B3 (Confidence) and B4 (Conf + Rel), ACARA-U exhibits net downward tier reclassifications (48 downgrades vs 9 upgrades in B3; 56 downgrades vs 8 upgrades in B4). This observed shift reflects the impact of the epistemic uncertainty penalty in the router ($\gamma = 1.0$) and in DCRI ($\delta = 0.10$), which lowers scores when uncertainty is high.
- **Observed Shifts vs B5**: Exactly three packets moved from Tier 0 under B5 to Tier 1 under B6 (with 0 downward shifts). This is an observed change associated with enabling the quality term ($\eta = 0.5$) in the evaluated configuration; clinical correctness was not established by this transition analysis alone.
