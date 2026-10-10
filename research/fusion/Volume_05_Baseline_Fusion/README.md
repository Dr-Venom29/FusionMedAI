# Volume 05 — Multimodal Baseline Ladder & Comparative Evaluation (Phase C11.5)

## Volume Overview & Scope

This volume presents the comprehensive empirical and comparative evaluation of the **ACARA-U Multimodal Decision Fusion Architecture** against a rigorous 6-tier baseline ladder in **FusionMedAI**:

$$\mathcal{B} = \{ \text{B1, B2, B3, B4, B5, B6} \}$$

Phase C11.5 investigates the question:

> **How does adaptive decision-level routing using confidence ($C_i$), global reliability ($R_i$), predictive uncertainty ($U_i$), input quality ($Q_i$), and availability ($A_i$) behave relative to simpler fusion heuristics under controlled availability and perturbation conditions?**

Evaluation is conducted over $N=500$ deterministic `CONTROLLED_DECISION_PACKET`s drawn from held-out validation prediction pools with zero cross-patient pairing claims and zero ground-truth label contamination.

**Status:** Phase C11.5 implementation, comparative benchmarking, and independent verification are complete and sealed (`18/18 gates passed`, `18/18 unit & behavioral tests passed`).

---

## Documents in this Volume

| Document | Focus Area | Key Formalizations |
| :--- | :--- | :--- |
| [`01_Experimental_Protocol.md`](./01_Experimental_Protocol.md) | Governance & Boundaries | Two-tier protocol, synthetic pairing prohibition, and decision-level scope declaration. |
| [`02_Baseline_Definitions.md`](./02_Baseline_Definitions.md) | Baseline Ladder Formalization | Exact mathematical formulas for B1 (Reliability-Selected), B2 (Uniform), B3 (Confidence), B4 (Conf+Rel), B5 (Conf+Rel+Uncertainty), and B6 (Full ACARA-U). |
| [`03_Decision_Packet_Construction.md`](./03_Decision_Packet_Construction.md) | Dataset & Sampling Protocol | Held-out validation prediction pool sizes, stratification, deterministic PRNG seed ($115$), and packet integrity invariants. |
| [`04_Behavioral_Metrics.md`](./04_Behavioral_Metrics.md) | Evaluation Metrics | Modality weight distribution, routing entropy $H(w)$, dominant modality frequency, and cross-modality risk divergence ($X_{ij}, X_{\max}$). |
| [`05_Missing_Modality_Evaluation.md`](./05_Missing_Modality_Evaluation.md) | Availability & Degradation | Full $7 \times 6 = 42$ operational matrix across all non-empty subsets of $\mathcal{M} = \{R, F, C\}$ and zero-modality handling. |
| [`06_Perturbation_Evaluation.md`](./06_Perturbation_Evaluation.md) | Step Response & Sensitivity | Controlled response curves under confidence sweep ($C \uparrow$), uncertainty surge ($U \uparrow$), and quality degradation ($Q \downarrow$). |
| [`07_Results.md`](./07_Results.md) | Empirical Benchmark Findings | Comprehensive cohort statistics ($N=500$), baseline ladder comparisons, ablation insights, and disagreement profiles. |
| [`08_Freeze_Report.md`](./08_Freeze_Report.md) | Formal Sign-Off & Handoff | Locked baseline comparison artifacts, gate compliance audit (18/18), and milestone handoff to Phase C11.6. |
| [`09_Outcome_Grounded_Evaluation_Addendum.md`](./09_Outcome_Grounded_Evaluation_Addendum.md) | Outcome-Grounded Evaluation | Paired predictive accuracy evaluation (B6 vs B5, $\Delta_{\mathrm{MAE}} = -0.001843$) across $N=5,000$ synthetic oracle packets. |
| [`10_Outcome_Evaluation_Protocol.md`](./10_Outcome_Evaluation_Protocol.md) | Outcome Evaluation Protocol | Full confirmatory protocol, hypotheses (H1–H5), synthetic oracle data generation, and artifact manifest. |

---

## Scientific Figures ([`./figures/`](./figures/))

| Figure | Description & Document Reference |
| :--- | :--- |
| [`baseline_weight_distribution.png`](./figures/baseline_weight_distribution.png) | Weight allocation $w_R, w_F, w_C$ across B1–B6 in [`07_Results.md`](./07_Results.md). |
| [`routing_entropy.png`](./figures/routing_entropy.png) | Mean routing entropy vs. theoretical max $\ln(3)$ in [`07_Results.md`](./07_Results.md). |
| [`modality_dominance.png`](./figures/modality_dominance.png) | Dominant modality plurality allocation rates in [`07_Results.md`](./07_Results.md). |
| [`missing_modality_behavior.png`](./figures/missing_modality_behavior.png) | ACARA-U authority redistribution across 7 operational configs in [`05_Missing_Modality_Evaluation.md`](./05_Missing_Modality_Evaluation.md). |
| [`perturbation_response.png`](./figures/perturbation_response.png) | Step-response sensitivity curves ($C \uparrow, U \uparrow, Q \uparrow$) in [`06_Perturbation_Evaluation.md`](./06_Perturbation_Evaluation.md). |
| [`conflict_disagreement.png`](./figures/conflict_disagreement.png) | Pairwise and maximal absolute risk divergence profile ($X_{ij}, X_{\max}$) in [`07_Results.md`](./07_Results.md). |

---

## Summary of the 6-Tier Baseline Ladder

| Baseline ID | Name | Scoring Rule / Logit $z_i$ | Normalization & Availability |
| :--- | :--- | :--- | :--- |
| **B1** | Reliability-Selected Unimodal | $i^* = \arg\max_{i \in \mathcal{A}} R_i$ | $w_{i^*} = 1.0, \quad w_{j \ne i^*} = 0.0$ |
| **B2** | Uniform Average | $w_i = \frac{1}{\vert\mathcal{A}\vert}$ | Direct uniform assignment over active set $\mathcal{A}$ |
| **B3** | Confidence-Only Fusion | $z_i = 1.0 \times C_i$ | Masked Softmax over active set $\mathcal{A}$ |
| **B4** | Confidence + Reliability | $z_i = 1.0 \times C_i + 1.0 \times R_i$ | Masked Softmax over active set $\mathcal{A}$ |
| **B5** | Conf + Rel + Uncertainty | $z_i = 1.0 \times C_i + 1.0 \times R_i - 1.0 \times U_i$ | Masked Softmax over active set $\mathcal{A}$ |
| **B6** | Full ACARA-U | $z_i = 1.0 \times C_i + 1.0 \times R_i - 1.0 \times U_i + 1.0 \times Q_i$ | Masked Softmax over active set $\mathcal{A}$ |

> [!NOTE]
> **Parameter Configuration Scope**: The baseline ladder table above displays the unit-coefficient formulation used in the initial Phase C11.5 behavioral routing exploration ($N=500$). The subsequent outcome-grounded confirmatory evaluation ([`09`](./09_Outcome_Grounded_Evaluation_Addendum.md), [`10`](./10_Outcome_Evaluation_Protocol.md)) evaluates B6 vs B5 using the separately frozen reference configuration $\Theta_0 = (\alpha=1.0, \beta=1.5, \gamma=1.0, \eta=0.5)$ and uncertainty discount $\delta^* = 0.10$ established across later research phases.

---

## Key Findings Snapshot ($N=500$ Controlled Cohort)

1. **Unimodal Insensitivity (B1)**: B1 is structurally insensitive to instance-level confidence, uncertainty, and quality because its selection rule depends solely on frozen historical modality reliability ($H(w) = 0.0000$), assigning $100.0\%$ weight to Retina across all tri-modal packets.
2. **Context-Insensitive Equal Weights (B2)**: Static $33.3\%$ per modality ($H(w) = 1.0986 = \ln 3$), treating all active channels equally regardless of individual certainty, predictive dispersion, or quality signals.
3. **Incremental Authority Realignment (B3 $\to$ B6)**: Dynamic authority balancing (Retina $47.7\%$, Foot $26.8\%$, Clinical $25.5\%$, mean $H(w) = 1.0176$), prioritizing reliable inputs while penalizing high predictive uncertainty and incorporating modality-specific input quality into routing authority.
4. **Graceful Sub-Network Fallback**: Under all single- and bi-modal dropouts, B6 redistributes $100\%$ weight across remaining valid modalities with strict conservation $\sum_{i \in \mathcal{A}} w_i = 1.0$.
