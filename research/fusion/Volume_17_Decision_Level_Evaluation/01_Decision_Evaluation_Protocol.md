# Phase C11.17: Final Decision-Level Evaluation Protocol

## 1. Executive Protocol Specification
This document establishes the frozen, pre-declared protocol for **Phase C11.17: Final Decision-Level Evaluation of ACARA-U** within the FusionMedAI clinical multimodal fusion framework.

- **Phase Objective**: Consolidate and establish the decision-level algorithmic performance, safety guarantees, tier distributions, and comparative characteristics of the full, frozen ACARA-U policy (Baseline B6) against established comparators (Baselines B1 through B5).
- **Core Principle**: Pure evaluation phase. No hyperparameter optimization, threshold alterations, or scoring equation tuning is permitted.

## 2. Frozen Architectural Invariants
1. **Decision Policy**: ACARA-U v2 frozen router with hyperparameters:
   - Dynamic Predictive Confidence Weight: $\alpha = 1.0$
   - Historical Modality Reliability Prior Weight: $\beta = 1.5$
   - Predictive Uncertainty Penalty Weight: $\gamma = 1.0$
   - Input Signal Quality Bonus Weight: $\eta = 0.5$
2. **Dual-Constraint Risk Index (DCRI)**:
   - Formally selected linear epistemic discount parameter: $\delta^* = 0.10$
   - Canonical unweighted formulation: $\text{DCRI} = R_{\text{fusion}} - \delta^* \sum_{i \in \mathcal{A}} U_i$
   - Negative DCRI Floor Policy: Strictly unclamped in metrics and reports to preserve full diagnostic transparency.
3. **Hypothetical Decision Tiers**:
   - Tier 0 (Routine Review): $\text{DCRI} < 0.20$
   - Tier 1 (Additional Assessment): $0.20 \le \text{DCRI} < 0.40$
   - Tier 2 (Escalation for Human Review): $\text{DCRI} \ge 0.40$
4. **Modality Availability Regimes**:
   - Full tri-modal set: $\mathcal{A} = \{R, F, C\}$ (Regime RFC)
   - Dual-modality combinations: $\{R, F\}, \{R, C\}, \{F, C\}$
   - Singleton regimes: $\{R\}, \{F\}, \{C\}$
   - Fail-Closed Empty Input: $\mathcal{A} = \emptyset \implies w_i = 0.0, R_{\text{fusion}} = 0.0, \text{DCRI} = 0.0$, `status = NO_MODALITY_AVAILABLE`.

## 3. Evaluation Cohort and Dataset Structure
- **Cohort Composition**: The evaluation cohort comprises exactly $N=500$ controlled multi-modal decision packets generated under deterministic seed `115` (`src/fusion/data/packet_generator.py`).
- **Packet Definition**: Each packet represents a synthetic clinical encounter containing three aligned diagnostic channels:
  1. *Retina*: Fundus photography model output (risk, confidence, uncertainty, quality, reliability).
  2. *Foot*: Thermographic/visual lesion model output (risk, confidence, uncertainty, quality, reliability).
  3. *Clinical*: Structured tabular electronic health record predictor output (risk, confidence, uncertainty, quality, reliability).
- **Outcome Labels**: This evaluation cohort evaluates algorithmic routing and policy behavior across controlled synthetic input distributions; it does not contain external clinical diagnostic ground-truth labels. Consequently, metrics evaluate policy concordance, tier reclassification, and mathematical invariant satisfaction rather than clinical diagnostic sensitivity or specificity.

## 4. Reproducibility & Statistical Settings
All numerical results are generated with deterministic reproducibility parameters:
- **Canonical Cohort Seed**: `DATA_SEED = 115` (sample size $N=500$).
- **Bootstrap Resampling**: `BOOTSTRAP_ROUNDS = 1,000` paired resamples.
- **Bootstrap Seed**: `BOOTSTRAP_SEED = 7117` (`numpy.random.default_rng(7117)`).
- **Nominal Significance Level**: `BOOTSTRAP_ALPHA = 0.05` ($95\%$ empirical percentile confidence intervals $[2.5\%, 97.5\%]$).
- **Pairing Unit**: Individual packet identifier (`packet_id`), ensuring exact packet-level pairing across comparator policies and degradation conditions.

## 5. Algorithmic Interpretation & Clinical Boundaries
1. **Nature of DCRI**: DCRI is an algorithmic decision index calculated from the frozen subtractive formula. It is not an empirical posterior disease probability. Because epistemic uncertainty is subtracted from fused risk, high model uncertainty reduces the numerical DCRI score, which can produce negative values when $\delta^* \sum U_i > R_{\text{fusion}}$.
2. **Clinical Validation Boundary**: Conforming to mathematical invariants, passing independent verifier acceptance gates, and maintaining bitwise checksum reproducibility verify software implementation correctness and protocol compliance. They do not establish clinical diagnostic efficacy, safety in uncurated patient cohorts, or regulatory deployment readiness.
3. **Protocol Freeze**: All thresholds ($\tau_1, \tau_2$), coefficients ($\alpha, \beta, \gamma, \eta$), discount factor ($\delta^*$), and evaluation pipelines are strictly locked. No post-hoc alterations are permitted.
