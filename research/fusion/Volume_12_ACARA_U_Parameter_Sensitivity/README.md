# Volume 12: ACARA-U Parameter & Weighting Sensitivity Analysis

## Executive Summary

Volume 12 provides a comprehensive sensitivity characterization of the ACARA-U dynamic multimodal router weighting coefficients $\Theta = (\alpha, \beta, \gamma, \eta)$ around the frozen reference configuration:

$$
\Theta_0 = (\alpha=1.00, \beta=1.50, \gamma=1.00, \eta=0.50)
$$

Evaluated across $N=500$ paired controlled decision packets under 23 pre-specified configuration evaluations representing 19 unique coefficient vectors (One-Factor-At-A-Time sweeps + combined factorial levels), this volume demonstrates that ACARA-U occupies a **tested, entropy-rich, and non-degenerate neighborhood** of the hyperparameter space.

---

## Volume Organization

- [`01_Protocol.md`](01_Protocol.md): Scientific context, pre-specified research questions, and 8 formal hypotheses.
- [`02_Parameter_Grid.md`](02_Parameter_Grid.md): Mathematical structure of the 23 named evaluations (19 unique parameter vectors) and exact logit derivative semantics.
- [`03_Reference_Configuration.md`](03_Reference_Configuration.md): Baseline performance and empirical profile of $\Theta_0$.
- [`04_Sensitivity_Results.md`](04_Sensitivity_Results.md): Complete OFAT empirical results, response dynamics, and linear/normalized slopes.
- [`05_Regime_Analysis.md`](05_Regime_Analysis.md): Evaluation across all 7 availability regimes and the empty modality set.
- [`06_Statistical_Analysis.md`](06_Statistical_Analysis.md): Reference-neighborhood comparisons and 1,000-resample paired bootstrap inference.
- [`07_Hypothesis_Results.md`](07_Hypothesis_Results.md): Detailed evidence and status for Hypotheses H1–H8.
- [`08_Methodological_Boundaries.md`](08_Methodological_Boundaries.md): Boundary declarations separating parameter sensitivity from optimization.
- [`09_Results_Scoreboard.md`](09_Results_Scoreboard.md): Unified summary table across all 23 evaluations.
- [`10_Freeze_Report.md`](10_Freeze_Report.md): Cryptographic SHA-256 manifest and formal sign-off.

---

## Primary Empirical Findings

1. **Local Stability**: Across all 23 tested $\pm 50\%$ coefficient perturbations on the $N=500$ packets ingested via `RetinaAdapter`, `FootAdapter`, and `ClinicalAdapter` into `EvidenceOrchestrator` and `ACARAUv2Router`, mean routing entropy remains high ($\bar{H} \in [1.0044, 1.0670]\text{ nats}$, with reference $\bar{H}(\Theta_0) = 1.038987\text{ nats}$ and mean weights $(\bar{w}_R, \bar{w}_F, \bar{w}_C) = (0.427135, 0.223079, 0.349785)$), preventing routing collapse or single-channel monopolization.
2. **Sensitivity Hierarchy (Normalized Aggregate Euclidean Slopes)**:
   - **Uncertainty Penalty ($\gamma$)** is the most responsive term ($S_\gamma^{\text{agg,norm}} = 0.424245$, linear norm $0.108461$).
   - **Reliability ($\beta$)** provides steady validation anchoring ($S_\beta^{\text{agg,norm}} = 0.121334$, linear norm $0.028654$).
   - **Confidence ($\alpha$)** provides dynamic per-case adaptation ($S_\alpha^{\text{agg,norm}} = 0.053949$, linear norm $0.015007$).
   - **Quality ($\eta$)** introduces gentle, non-disruptive bonus modulation ($S_\eta^{\text{agg,norm}} = 0.034830$, linear norm $0.026745$).
3. **Reference Retained**: The frozen reference configuration $\Theta_0 = (1.0, 1.5, 1.0, 0.5)$ is retained without modification for Phase C11.13.

