# FusionMedAI

> An interpretable, uncertainty-aware framework for adaptive decision-level routing of heterogeneous medical AI evidence.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch 2.4](https://img.shields.io/badge/pytorch-2.4-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. FusionMedAI: Overview and Research Contributions

### Abstract
FusionMedAI investigates how independently trained retinal, diabetic foot-ulcer, and structured clinical prediction models can be orchestrated through **ACARA-U** (*Adaptive Confidence-, Reliability-, and Uncertainty-Aware routing*). The framework combines sample-level confidence, validation-derived reliability, predictive uncertainty, and input quality to allocate review-priority authority across available modalities, with explicit constraints for missing evidence and cross-modality disagreement.
Rather than merging incompatible prediction targets into a purported unified clinical risk, FusionMedAI preserves each modality's native output and evaluates historical scalar fusion indices separately as controlled research benchmarks.

### Core Research Contributions
1. **Decision-Level Routing Kernel (ACARA-U)**: A closed-form, interpretable routing logit formulation ($z_i = \alpha C_i + \beta R_i - \gamma U_i + \eta Q_i$) that dynamically allocates review-priority weights over the active modality simplex while guaranteeing zero authority to missing sensors via hard masking.
2. **Orthogonal Downstream Uncertainty Penalization (DCRI)**: An unweighted cumulative uncertainty discount ($P_U = \delta \sum_{i \in \mathcal{A}} U_i$) applied to fused summary scores, rigorously decoupled from upstream routing dynamics ($\Delta w \equiv 0$).
3. **Controlled Multi-Dimension Benchmark Suite**: Comprehensive empirical stress testing across baseline comparator ladders (B1–B6), 8 modality availability regimes, 12 signal degradation operators (D0–D3), cross-modality conflict stratification, and multi-cohort tail-sensitivity distributions.
4. **Transparent Scientific Boundary Characterization**: Rigorous negative finding disclosures, establishing that quality-aware routing provides conditional error reduction under severe synthetic degradation while practical superiority under clean data and multi-cohort tail distributions remains unestablished.

---

## 2. System Architecture and Fusion Methodology

### System Architecture Flow

```mermaid
flowchart TD
    subgraph Inputs["1. Diagnostic Modality Inputs (Independent Contexts)"]
        R_in["Fundus Photography<br/>(APTOS-2019 Scans)"]
        F_in["Wound Photographs<br/>(DFU ADPM V3.3)"]
        C_in["Structured EHR Records<br/>(UCI 130-Hospitals, D=119)"]
    end
    subgraph Modalities["2. Independent Modality Backbones & Calibration"]
        R_model["EfficientNet-B3 (QWK=0.9233)<br/>Temperature Scaling (ECE=0.0241)<br/>25-Pass MC Dropout Uncertainty"]
        F_model["EfficientNet-B3 (Macro F1=0.6683)<br/>Vector Scaling (ECE=0.0313)<br/>10-Pass MC Dropout Uncertainty"]
        C_model["CatBoost HPO (ROC-AUC=0.6494)<br/>Isotonic Calibration (ECE=0.0032)<br/>20-Member Bootstrap Uncertainty"]
    end

    Inputs --> Modalities
    subgraph Adapters["3. Standardized Modality Evidence Adapters"]
        R_pack["Retina Evidence Contract:<br/>(r_R, p_cal, C_R, U_R, Q_R, A_R, R_R)"]
        F_pack["Foot Evidence Contract:<br/>(r_F, p_cal, C_F, U_F, Q_F, A_F, R_F)"]
        C_pack["Clinical Evidence Contract:<br/>(r_C, p_cal, C_C, U_C, Q_C, A_C, R_C)"]
    end
    Modalities --> Adapters
    subgraph Orchestrator["4. EvidenceOrchestrator & Execution Boundary"]
        Orch["Canonical Packet Manifest (N=500)<br/>Hard Availability Masking: A_i = 0 ⇒ z_i = -∞<br/>Fail-Closed Gating: |A| = 0 ⇒ HALT"]
    end
    Adapters --> Orchestrator
    subgraph Router["5. ACARA-U Dynamic Authority Router"]
        Logit["Logit Kernel: z_i = α C_i + β R_i - γ U_i + η Q_i<br/>Frozen Reference: Θ_0 = (1.0, 1.5, 1.0, 0.5)"]
        Softmax["Stable Simplex Softmax: w_i = exp(z_i - m) / Σ exp(z_j - m)<br/>Conservation Invariant: Σ w_i = 1.0"]
        Logit --> Softmax
    end
    Orchestrator --> Router

    subgraph Aggregation["6. Aggregation & Research Metric Engine"]
        NativeOut["Preserved Native Outputs<br/>(Retina Grade, Wagner Class, Readmission Risk)"]
        FusedRisk["Base Fused Risk Scalar:<br/>R_fusion = Σ w_i r_i  ∈ [0, 1]"]
        Conflict["Continuous Discordance:<br/>CSS, Δ_max, σ_w"]
        DCRI["DCRI Epistemic Penalty:<br/>P_U = δ Σ U_i (Selected δ* = 0.10)<br/>DCRI_0.10 = R_fusion - P_U ∈ [-0.30, 1.0]"]
        Softmax --> FusedRisk
        Softmax --> Conflict
        FusedRisk --> DCRI
    end
    Router --> NativeOut
    subgraph PolicyLayer["7. Decision-Policy Layer & Safeguard Gateway"]
        direction TB
        Gate{"Policy S Safeguard Gateway<br/>Is Ambiguous?<br/>(Σ U_i ≥ 0.30  OR  min Q_i < 0.60)"}
        DCRI --> Gate
        FusedRisk --> Gate
        
        SubUnconstrained["Policy A: Unconstrained DCRI<br/>Direct DCRI_0.10 Mapping"]
        SubBase["Policy B: Base Fused Risk<br/>Direct R_fusion Mapping"]
        SubSafeguard["Policy S: Safeguarded DCRI<br/>If Ambiguous & Downgrade:<br/>Hold Base Tier + Emit Flag"]
        
        Gate --> SubSafeguard
        DCRI -.-> SubUnconstrained
        FusedRisk -.-> SubBase
        
        Tiers["Action Triage Queues:<br/>• Tier 0: Routine Review (&lt; 0.20)<br/>• Tier 1: Additional Assessment [0.20, 0.40)<br/>• Tier 2: Priority Escalation (≥ 0.40)"]
        FlagNode["Workflow Action Flag:<br/>FLAG_REPEAT_EXAM_AMBIGUOUS_EVIDENCE"]

        SubSafeguard --> Tiers
        SubSafeguard --> FlagNode
    end

```

### End-to-End Pipeline Architecture Layers
The FusionMedAI orchestration pipeline operates across six distinct functional layers:
1. **Layer 1: Modality Backbones & Calibrated Risk Projections**:
   - Independently trained diagnostic backbones process distinct clinical modalities (APTOS fundus images, ADPM wound photos, UCI EHR records).
   - In-sample probability calibration (Temperature Scaling for Retina, Vector Scaling for Foot, Isotonic Regression for Clinical) ensures reliable risk projections $r_i \in [0, 1]$.
   - Epistemic predictive uncertainty $U_i \in [0, 1]$ is quantified via Monte Carlo Dropout (imaging) or Bootstrap ensembles (tabular).
2. **Layer 2: Standardized Evidence Contracts & Live Quality Engines**:
   - Each channel encapsulates its state into an immutable 8-tuple contract:
     $$\mathcal{M}_i = \langle r_i, p_{i,\text{cal}}, C_i, U_i, Q_i, A_i, R_i, \text{ver}_i \rangle$$
   - Unsupervised quality engines ($Q_i$) assess signal integrity in real time (e.g., BRISQUE for fundus sharpness, CNR for wound illumination, feature missingness for EHR).
   - Validation-derived reliability priors ($R_i = \frac{1}{2}[\text{AUC}_i + (1 - \text{ECE}_i)]$) anchor channels based on verified predictive track records.
3. **Layer 3: Evidence Orchestrator & Availability Enforcement**:
   - Manages packet ingestion, schemas, and missing-modality safety.
   - Enforces hard availability masking: channels with $A_i = 0$ are assigned logits $z_i = -\infty$, guaranteeing exactly $w_i = 0.000000$ with zero numerical leakage.
   - Implements fail-closed halting when no modalities are available ($|\mathcal{A}| = 0 \implies \texttt{NO\_MODALITY\_AVAILABLE}$).
4. **Layer 4: ACARA-U Dynamic Authority Router**:
   - Computes dynamic routing logits combining all four evidence signals:
     $$z_i = \alpha C_i + \beta R_i - \gamma U_i + \eta Q_i \quad (\Theta_0 = (1.0, 1.5, 1.0, 0.5))$$
   - Allocates authority weights $w_i$ over the active simplex via numerically stable softmax ($\sum_{i \in \mathcal{A}} w_i = 1.0$).
5. **Layer 5: Continuous Risk Aggregation & DCRI Uncertainty Discounting**:
   - Calculates base fused review risk as the authority-weighted sum: $R_{\mathrm{fusion}} = \sum_{i \in \mathcal{A}} w_i r_i$.
   - Computes continuous inter-sensor discordance metrics ($\Delta_{\max}, \sigma_w, \text{CSS}$) to detect diagnostic conflict.
   - Derives the Dynamic Clinical Risk Index ($\text{DCRI}_{0.10} = R_{\mathrm{fusion}} - 0.10 \sum_{i \in \mathcal{A}} U_i$), orthogonally discounting the summary score by cumulative evidential uncertainty without altering upstream routing weights ($\Delta w \equiv 0$).
6. **Layer 6: Decision-Policy Safeguard Gateway & Action Triage**:
   - Maps continuous scores to action tiers ($\tau_1 = 0.20, \tau_2 = 0.40$): Routine Review (Tier 0), Additional Assessment (Tier 1), Priority Escalation (Tier 2).
   - Implements the **Policy S Ambiguity Gateway**: evaluates whether $\sum U_i \ge 0.30$ or $\min(Q_i) < 0.60$.
   - If ambiguous, suppresses downward tier reclassifications ($T_{\mathrm{final}} = T_{\mathrm{base}}$) to eliminate false downgrades, while emitting `FLAG_REPEAT_EXAM_AMBIGUOUS_EVIDENCE` to route uncertain patients for diagnostic re-testing.

---

### Comparative Evaluation of Evidence-Processing Strategies
To evaluate the contribution of adapter-level evidence normalization and orchestration while isolating confidence-definition semantics, FusionMedAI establishes three evaluation tiers operating upstream of the same ACARA-U routing engine ($\Theta_0 = (1.0, 1.5, 1.0, 0.5)$):
1. **Method A — Direct-Packet Routing Baseline (Controlled Baseline)**:
   - Evaluates decision routing and DCRI using standardized packets constructed directly without intermediate adapter transformation or orchestration gating, holding confidence definition, model probabilities, and inputs constant to isolate pipeline overhead.
   - When inputs and confidence semantics are held identical, routing calculations on valid packets produce exact machine-precision numerical agreement ($\max \lvert\Delta w\rvert < 5.0 \times 10^{-7}$). The primary contribution of the canonical pipeline is therefore architectural safety, strict type invariants, and fail-closed availability gating rather than numerical deviation on clean inputs.
2. **Method B — Canonical Evidence-Pipeline Routing (Primary)**:
   - Evaluates the complete multi-stage pipeline: $\text{Modality Adapters} \longrightarrow \text{EvidenceOrchestrator} \longrightarrow \text{ACARA-U Router} \longrightarrow \text{DCRI Evaluation}$.
   - Strictly enforces semantic evidence contracts, fail-closed safety assertions, and consistent decision-boundary confidence margins ($C_{\text{clinical}} = 2\lvert\hat{p} - 0.5\rvert$, mean $0.776$).
3. **Method C — Confidence-Definition Sensitivity Comparison (Heuristic Comparator)**:
   - Deliberately tests the alternative heuristic confidence definition ($C_{\text{alt}} = 1 - 2\lvert\hat{p} - 0.5\rvert$, mean $0.224$, where $C_{\text{alt}} + C_{\text{canonical}} = 1.0$).
   - Because confidence enters the logit kernel directly ($z_i = \alpha C_i + \beta R_i - \gamma U_i + \eta Q_i$), this semantic shift accounts for the observed authority shift between historical unnormalized packets and the canonical pipeline.

#### Strategy and Sensitivity Comparison Summary ($N=500$, seed 115, $\Theta_0$)

| Evaluation Dimension | Method A: Direct Baseline (Canonical $C$) | Method B: Canonical Pipeline (Primary) | Method C: Heuristic Confidence ($C_{\text{alt}}$) | Methodological Impact & Attribution |
| :--- | :---: | :---: | :---: | :--- |
| **Retina Weight ($w_R$)** | **$42.71\% \pm 7.88\%$** | **$42.71\% \pm 7.88\%$** | $50.10\% \pm 8.83\%$ | Method A & B identical; Method C shifts $+7.39\%$ due to low clinical $C$ |
| **Foot Ulcer Weight ($w_F$)** | **$22.31\% \pm 7.72\%$** | **$22.31\% \pm 7.72\%$** | $26.09\% \pm 8.64\%$ | Method A & B identical; Method C shifts $+3.78\%$ |
| **Clinical EHR Weight ($w_C$)** | **$34.98\% \pm 6.50\%$** | **$34.98\% \pm 6.50\%$** | $23.81\% \pm 5.23\%$ | Canonical boundary confidence increases tabular authority ($+11.17\%$) |
| **Routing Shannon Entropy** | **$1.038987\text{ nats}$** | **$1.038987\text{ nats}$** | $1.011041\text{ nats}$ | Canonical confidence yields higher allocation balance ($+0.027946\text{ nats}$) |
| **Retina Dominance Rate** | **$73.2\%\ (366 / 500)$** | **$73.2\%\ (366 / 500)$** | $91.2\%\ (456 / 500)$ | Canonical confidence avoids excessive single-channel concentration |
| **Mean Fused Risk ($\overline{R}$)** | **$0.261017$** | **$0.261017$** | $0.289900$ | Difference between B and C driven entirely by clinical confidence definition |
| **Mean DCRI ($\overline{\text{DCRI}}_{0.10}$)** | **$0.197624$** | **$0.197624$** | $0.226506$ | Proportional shift preserving unweighted penalty ($P_U = 0.063394$) |
| **Negative DCRI Count** | **$38\ (7.6\%)$** | **$38\ (7.6\%)$** | $37\ (7.4\%)$ | Stable bounded negative frequency ($\le 10\%$ criterion met across all) |
| **Action Tier Distributions** | **$52.4\% / 36.2\% / 11.4\%$** | **$52.4\% / 36.2\% / 11.4\%$** | $46.4\% / 37.4\% / 16.2\%$ | Method A and B match; Method C assigns more cases to Escalation |
| **Contract & Invariant Gating** | Permissive / Unvalidated | **Enforced / Fail-Closed** | Permissive / Unvalidated | Primary contribution of Method B is architectural safety and invariant checks |

*Methodological Finding*: When confidence semantics and inputs are held constant (Method A vs Method B), routing outputs match to within floating-point tolerance ($\Delta w < 5.0 \times 10^{-7}$). The numeric authority reallocation is entirely explained by the confidence semantic definition (Method C vs Method B), while Method B provides automated invariant gating, semantic typing, and fail-closed availability enforcement.

### Target Preservation & Rejection of Common-Target Collapse
FusionMedAI explicitly rejects the practice of forcing clinically heterogeneous prediction targets (e.g., retinal microvascular damage, Wagner foot ulcer grade, and 30-day hospital readmission) into a single scalar "universal clinical risk":
- **Preserved Native Outputs**: In clinical deployment, each modality's discrete prediction and native target must be preserved and presented in its native clinical framing.
- **Role of Fused Indices**: Scalar aggregations ($R_{\text{fusion}}, \text{DCRI}$) are evaluated strictly as **review-prioritization and triage indices** for resource allocation, not as unified biological probabilities.

---

## 3. Modality-Specific Models and Independent Evaluation

Each modality backbone was evaluated on held-out test partitions under strict leakage-prevention controls and frozen contracts:

| Modality & Dataset | Architecture & Params | Primary Performance | Probability Calibration | Uncertainty Quantification | Explainability | Research Volume |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Retina (Fundus)**<br>APTOS 2019 ($N_{\text{test}}=367$) | EfficientNet-B3<br>($10.70\text{M}$ params) | **$\text{QWK} = 0.9233$**<br>$\text{Acc} = 84.20\%$ | Temperature Scaling<br>$\text{ECE} = 0.0241$ | 25-pass MC Dropout<br>$\text{Error AUROC} = 0.8443$ | Spatial Grad-CAM<br>(Microaneurysms) | [Retina Volume 05](research/retina/Volume_05_Architecture_Benchmarking/README.md) |
| **Foot Ulcer (Wound)**<br>ADPM V3.3 ($N_{\text{test}}=1,006$) | EfficientNet-B3<br>($10.70\text{M}$ params) | **$\text{Macro F1} = 0.6683$**<br>$\text{Macro AUC} = 0.8685$ | Vector Scaling<br>$\text{ECE} = 0.0313$ | 10-pass MC Dropout<br>$\text{Entropy AUROC} = 0.7291$ | Spatial Grad-CAM<br>(Wound margins) | [Foot Volume 05](research/foot/Volume_05_Architecture_Benchmarking/README.md) |
| **Clinical Tabular (EHR)**<br>UCI 130-Hosp ($N_{\text{test}}=14,913$) | CatBoost HPO<br>($D=119$ features) | **$\text{ROC-AUC} = 0.6494$**<br>$\text{PR-AUC} = 0.2063$ | Isotonic Regression<br>$\text{ECE} = 0.0032$ | 20-member Bootstrap<br>$\text{Error AUROC} = 0.7116$ | Exact TreeSHAP<br>($\rho = 0.9994$) | [Clinical Volume 05](research/clinical/Volume_05_Architecture_Benchmarking/README.md) |

### Modality Contract Fields & Reliability Priors
Each modality emits an 8-tuple contract $\mathcal{M}_i = (r_i, p_{i,\text{cal}}, C_i, U_i, Q_i, A_i, R_i, \text{ver}_i)$.
The static reliability prior $R_i = \frac{1}{2}[\text{AUC}_i + (1 - \text{ECE}_i)]$ scales each channel's baseline authority using frozen held-out validation evidence:
- **Retina ($R_R = 0.929956$)**: Validation Macro OvR $\text{AUC} = 0.911149$, Validation 10-bin equal-frequency $\text{ECE} = 0.051237$ ($N_{\text{val}} = 366$).
- **Foot Ulcer ($R_F = 0.922266$)**: Validation Macro OvR $\text{AUC} = 0.884435$, Validation 10-bin equal-frequency $\text{ECE} = 0.039903$ ($N_{\text{val}} = 1,006$).
- **Clinical EHR ($R_C = 0.825382$)**: Validation Binary $\text{AUC} = 0.650764$, in-sample validation isotonic $\text{ECE} = 0.000000$ ($N_{\text{val}} = 14,911$).

> [!NOTE]
> **Clinical Reliability Prior Provenance & Sensitivity**:
> - **Data Split & Calibration Provenance**: For the Clinical EHR model, non-parametric Isotonic Regression was fitted on the validation set ($N_{\text{val}}=14,911$) to generate calibrated probabilities. Under the pre-specified protocol evaluated on that same split with 10 equal-frequency quantile bins, empirical accuracy aligns with bin confidence, producing an observed in-sample $\text{ECE} = 0.000000 \implies R_C = 0.825382$.
> - **Out-of-Sample Reference**: Evaluating parametric Platt Scaling on the validation split yields out-of-sample $\text{ECE} = 0.0030 \implies R_C = 0.823882$. Evaluating parametric Beta Calibration yields validation $\text{ECE} = 0.0040 \implies R_C = 0.823382$. On the locked held-out test split ($N_{\text{test}}=14,913$), Isotonic ECE is $0.0032 \implies R_C = 0.823782$.
> - **Sensitivity Impact**: The variation between the in-sample protocol value ($0.825382$) and out-of-sample calibration estimates ($0.823382\text{--}0.823882$) is bounded by $\Delta R_C \le 0.0020$ ($< 0.25\%$), producing a negligible shift in downstream routing logits ($\beta \cdot \Delta R_C \le 0.0030$). The pre-registered protocol value $R_C = 0.825382$ is retained for consistency while documenting this in-sample property transparently.
> [!NOTE]
> **Clinical Confidence Semantics**:
> Clinical confidence is defined as the decision-boundary margin:
> $$C_{\text{clinical}} = 2 \lvert \hat{p} - 0.5 \rvert = \lvert p_1 - p_0 \rvert$$
> This metric ranges in $[0, 1]$, where values near $1.0$ indicate high certainty in predicting either readmission or non-readmission. Because baseline 30-day readmission prevalence is low ($\sim 11\%$), confident non-readmission predictions ($\hat{p} \approx 0.10$) produce high margin confidence ($C_{\text{clinical}} \approx 0.80$). This reflects distance from the decision boundary, not posterior probability of a positive outcome.

---

## 4. ACARA-U: Evidence Routing and Authority Weights

### Routing Kernel Formulation
For each available modality $i \in \mathcal{A}$, the unnormalized routing logit $z_i$ is computed as:
$$z_i = \alpha C_i + \beta R_i - \gamma U_i + \eta Q_i$$
with frozen reference coefficients $\Theta_0 = (\alpha=1.0, \beta=1.5, \gamma=1.0, \eta=0.5)$.
To strictly prevent authority leakage to missing sensors, hard availability masking is applied:
$$\tilde{z}_i = \begin{cases} z_i & \text{if } A_i = 1 \\ -\infty & \text{if } A_i = 0 \end{cases}$$
Routing authority weights $w_i \in [0, 1]$ are assigned using stable softmax over active channels:
$$w_i = \frac{\exp(\tilde{z}_i - \max_{j \in \mathcal{A}} \tilde{z}_j)}{\sum_{k \in \mathcal{A}} \exp(\tilde{z}_k - \max_{j \in \mathcal{A}} \tilde{z}_j)}$$

### Canonical Routing Invariants & Distributions ($N=500$, seed 115)
Evaluated on the canonical manifest (`cf877d74e306...`):

| Routing Metric | Canonical Measured Value | Traceability / Artifact Link |
| :--- | :---: | :--- |
| **Mean Authority Weights ($w_R / w_F / w_C$)** | **$42.71\% \pm 7.88\% / 22.31\% \pm 7.72\% / 34.98\% \pm 6.50\%$** | [`03_router_intermediate_comparison.json`](experiments/fusion/canonical_redesigned_rerun/03_router_intermediate_comparison.json#L20-L25) |
| **Routing Entropy ($\bar{H}(w)$)** | **$1.038987 \pm 0.040595\text{ nats}$** (Range: $[0.8979, 1.0984]$) | [`03_router_intermediate_comparison.json`](experiments/fusion/canonical_redesigned_rerun/03_router_intermediate_comparison.json#L26) |
| **Dominant Modality Counts** | **Retina: 366 (73.2%), Clinical: 110 (22.0%), Foot: 24 (4.8%)** | [`03_router_intermediate_comparison.json`](experiments/fusion/canonical_redesigned_rerun/03_router_intermediate_comparison.json#L28-L32) |
| **Simplex Preservation** | **$\sum_{i \in \mathcal{A}} w_i = 1.000000 \pm 1.11 \times 10^{-16}$** | [`test_evidence_orchestration.py`](verification/fusion/contract/test_evidence_orchestration.py) |
| **Missing-Modality Invariant** | **$w_i \equiv 0.000000$ for all $A_i = 0$ ($0$ violations / $7,500$ trials)** | [`Volume 08 Missing Modality`](research/fusion/Volume_08_Missing_Modality_Robustness/README.md) |
| **Fail-Closed Degeneracy** | **$\lvert\mathcal{A}\rvert = 0 \implies \texttt{NO\_MODALITY\_AVAILABLE}$ ($R=0.0$)** | [`Volume 17 Decision Evaluation`](research/fusion/Volume_17_Decision_Level_Evaluation/README.md) |

### Comparative Baseline Ladder (B1–B6)
Evaluated across six pre-registered allocation policies on the canonical $N=500$ cohort:

| Baseline ID | Strategy Name | Routing Logit Formulation | Mean Fused Risk $\overline{R}$ | Mean DCRI $\overline{\text{DCRI}}$ | Routing Entropy $\bar{H}$ | Negative DCRI Count |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **B1** | Reliability-Selected | $w_{\text{retina}} \equiv 1.0$ (Unimodal) | $0.255037$ | $0.191643$ | $0.0000$ | $223\ (44.6\%)$ |
| **B2** | Uniform Average | $w_i = 1 / \lvert\mathcal{A}\rvert$ | $0.296804$ | $0.233410$ | $1.0986$ | $0\ (0.0\%)$ |
| **B3** | Confidence Fusion | $z_i = 1.0 C_i$ | $0.282063$ | $0.218670$ | $1.0841$ | $0\ (0.0\%)$ |
| **B4** | Conf. + Reliability | $z_i = 1.0 C_i + 1.5 R_i$ | $0.287679$ | $0.224286$ | $1.0843$ | $0\ (0.0\%)$ |
| **B5** | Conf + Rel - Uncertainty | $z_i = 1.0 C_i + 1.5 R_i - 1.0 U_i$ | $0.257236$ | $0.193843$ | $1.0403$ | $36\ (7.2\%)$ |
| **B6** | **Full ACARA-U** | $z_i = 1.0 C_i + 1.5 R_i - 1.0 U_i + 0.5 Q_i$ | **$0.261017$** | **$0.197624$** | **$1.0390$** | **$38\ (7.6\%)$** |

*Artifact Traceability*: [`04_canonical_c11_full_metrics.json`](experiments/fusion/canonical_redesigned_rerun/04_canonical_c11_full_metrics.json#L1826-L2032).

---

## 5. DCRI: Research Index and Parameter Selection

### Mathematical Definition & Domain
The Dynamic Clinical Risk Index (DCRI) calculates an uncertainty-discounted continuous research index by applying an additive unweighted penalty for cumulative epistemic uncertainty to the authority-weighted fused risk:
$$\begin{aligned}
R_{\text{fusion}} &= \sum_{i \in \mathcal{A}} w_i r_i \\
P_U(\delta) &= \delta \sum_{i \in \mathcal{A}} U_i = \delta U_{\text{sum}} \\
\text{DCRI}_\delta &= R_{\text{fusion}} - P_U(\delta) = \sum_{i \in \mathcal{A}} w_i r_i - \delta \sum_{i \in \mathcal{A}} U_i
\end{aligned}$$

- **Continuous Theoretical Domain**: Because $R_{\text{fusion}} \in [0, 1]$ and each active uncertainty $U_i \in [0, 1]$, the theoretical domain spans $[-\delta \lvert\mathcal{A}\rvert, 1.0]$. For the frozen setting $\delta^* = 0.10$ across up to three modalities ($\lvert\mathcal{A}\rvert \le 3$), the numerical domain is $[-0.30, 1.0]$.
- **Unclamped Negative Score Mechanics**: Negative DCRI occurs when cumulative uncertainty penalty strictly exceeds base fused risk ($P_U > R_{\text{fusion}}$). Negative scores are **intentionally left unclamped** rather than truncated to $0.0$. In research analysis, negative scores preserve rank ordering, maintain mathematical continuity, and provide an unclipped diagnostic signal that the evidence is too epistemically opaque to substantiate a positive risk assertion.

### Double-Use of Uncertainty in Architecture
Predictive uncertainty ($U_i$) serves two distinct, decoupled roles within the FusionMedAI framework:
1. **Relative Authority Modulation (Upstream Router)**:
   In the ACARA-U routing kernel, $-\gamma U_i$ acts as an unnormalized logit penalty. Modalities with higher predictive uncertainty surrender simplex authority to lower-uncertainty peers, redistributing relative weights ($w_i$).
2. **Absolute Evidential Discounting (Downstream Aggregation)**:
   In DCRI risk aggregation, $-\delta \sum U_i$ applies an unweighted additive penalty to the summary score. This reflects the total unobserved epistemic ignorance of the multi-sensor system, discounting confidence in the fused scalar regardless of which modality dominated the weights.

### Core Mathematical Invariants of DCRI
- **Monotonic Non-Inflation Invariant**: $\forall \delta \ge 0, U_i \ge 0 \implies \text{DCRI}_\delta \le R_{\text{fusion}}$. Epistemic discounting can only reduce or maintain the summary score; it is mathematically impossible for DCRI to inflate risk above base projections.
- **Strict Downstream Orthogonality**: Varying the penalty factor $\delta$ alters downstream $\text{DCRI}$ with exact zero feedback to upstream routing: $\Delta w_i \equiv 0.000000$ and $\Delta R_{\text{fusion}} \equiv 0.000000$ across all candidate values.
- **Exact Analytical Gradient**: The sensitivity derivative $\frac{\partial \overline{\text{DCRI}}}{\partial \delta} = -\overline{U_{\text{sum}}} = -0.633936$ holds with machine-precision zero residual ($< 10^{-16}$) across all candidate intervals.

### Rationale and Multi-Criteria Selection for $\delta^* = 0.10$
The research discount factor $\delta^* = 0.10$ was frozen via a pre-registered 11-candidate sweep ($\delta \in [0.00, 1.00]$) evaluated against three pre-declared criteria:
1. **Bounded Negative Rate**: Keep negative DCRI occurrences under $10\%$ on the tri-modal cohort.
2. **Rank Preservation**: Maintain rank fidelity with base fused risk (Spearman $\rho_s > 0.95$).
3. **Meaningful Penalty Discrimination**: Ensure mean penalty $P_U$ constitutes $15\%\text{--}30\%$ of mean base risk.

### Candidate Grid Evaluation ($N=500$, canonical rerun):

| Candidate ID | Discount Factor $\delta$ | Mean DCRI $\overline{\text{DCRI}}$ | Negative Rate (%) | Spearman Rank $\rho_s$ | Selection Status & Boundary |
| :---: | :---: | :---: | :---: | :---: | :--- |
| `D00` | $0.00$ | $0.261017$ | $0.0\%$ | $1.000000$ | Unpenalized baseline ($P_U \equiv 0$) |
| `D05` | $0.05$ | $0.229320$ | $0.0\%$ | $0.996144$ | Insufficient penalty discrimination |
| **`D10`** | **$0.10$** | **$0.197624$** | **$7.6\%$ ($38/500$)** | **$0.985623$** | **SELECTED RESEARCH OPERATING POINT** |
| `D15` | $0.15$ | $0.165927$ | $15.8\%$ | $0.969854$ | Exceeds 10% negative threshold |
| `D20` | $0.20$ | $0.134230$ | $25.4\%$ ($127/500$) | $0.949821$ | Historical provisional setting (excessive penalty) |
| `D50` | $0.50$ | $-0.055951$ | $67.4\%$ ($337/500$) | $0.825694$ | Severe rank inversion; negative mean score |

*Verification & Confirmation*: Under the canonical rerun, `D10` ($\delta = 0.10$) uniquely satisfies all three pre-registered acceptance criteria (negative rate $7.6\% \le 10\%$, rank preservation $\rho_s = 0.985623 > 0.95$, penalty ratio $P_U / \overline{R} = 24.28\% \in [15\%, 30\%]$). The selection remains fully supported under the canonical evaluation path.

### Regime Scaling Behavior Across Modality Permutations ($N=500$)
Because the penalty is additive across active channels ($P_U = 0.10 \sum_{i \in \mathcal{A}} U_i$), negative DCRI rates scale predictably with modality cardinality $\lvert\mathcal{A}\rvert$:

| Active Regime | Active Channels $\mathcal{A}$ | Cardinality $M$ | Mean Fused Risk $\overline{R}$ | Mean Penalty $\overline{P}_U$ | Mean $\overline{\text{DCRI}}_{0.10}$ | Negative DCRI Count (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RFC** | $\{R, F, C\}$ | 3 | $0.261017$ | $0.063394$ | $0.197624$ | $38\ (7.6\%)$ |
| **RF** | $\{R, F\}$ | 2 | $0.297424$ | $0.044769$ | $0.252655$ | $18\ (3.6\%)$ |
| **RC** | $\{R, C\}$ | 2 | $0.256940$ | $0.040524$ | $0.216416$ | $13\ (2.6\%)$ |
| **FC** | $\{F, C\}$ | 2 | $0.316447$ | $0.040239$ | $0.276208$ | $7\ (1.4\%)$ |
| **R** | $\{R\}$ | 1 | $0.255037$ | $0.000275$ | $0.254762$ | $15\ (3.0\%)$ |
| **F** | $\{F\}$ | 1 | $0.521907$ | $0.118783$ | $0.403124$ | $6\ (1.2\%)$ |
| **C** | $\{C\}$ | 1 | $0.113468$ | $0.007729$ | $0.105739$ | $4\ (0.8\%)$ |
| **EMPTY** | $\emptyset$ | 0 | $0.000000$ | $0.000000$ | $0.000000$ | $0\ (0.0\%)$ |

*Decoupling from Decision Policy*: While DCRI produces a mathematically sound continuous research score, Section 6 demonstrates that unconstrained subtraction of uncertainty can lead to under-triage of high-risk cases. This justifies separating the DCRI continuous calculation from downstream triage via the Policy S Ambiguity Gateway.
*Artifact Traceability*: [`04_canonical_c11_full_metrics.json`](experiments/fusion/canonical_redesigned_rerun/04_canonical_c11_full_metrics.json#L1437-L1542); [Volume 06 DCRI Aggregation](research/fusion/Volume_06_DCRI_Aggregation/README.md); [Volume 13 DCRI Delta Selection](research/fusion/Volume_13_DCRI_Delta_Selection/README.md).

---

## 6. Decision-Policy Analysis: Base Risk vs DCRI vs Safeguarded Policy S
To evaluate how uncertainty discounting and safety gating influence clinical resource allocation, we evaluate a hypothetical 3-tier action policy on the canonical cohort ($N=500$, seed 115) with standard operating thresholds $\tau_1 = 0.20, \tau_2 = 0.40$:
- **Tier 0 (Routine Review)**: $\text{Score} < 0.20$ (Primary care / standard surveillance queue)
- **Tier 1 (Additional Assessment)**: $0.20 \le \text{Score} < 0.40$ (Secondary diagnostic inspection / intermediate queue)
- **Tier 2 (Priority Escalation)**: $\text{Score} \ge 0.40$ (Urgent specialist consultation queue)

### Three-Policy Master Comparison ($N=500$, Canonical Cohort)

| Policy Formulation | Routine Review (Tier 0) | Additional Assessment (Tier 1) | Priority Escalation (Tier 2) | Total Downgrades vs Base | Blocked Downgrades | Repeat-Exam Flags Emitted | Operational Triage Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Policy B: Base Fused Risk ($R_{\text{fusion}}$)** | $192\ (38.4\%)$ | $225\ (45.0\%)$ | $83\ (16.6\%)$ | — | — | — | Unpenalized evidence reference |
| **Policy A: Unconstrained $\text{DCRI}_{0.10}$** | **$262\ (52.4\%)$** | **$181\ (36.2\%)$** | **$57\ (11.4\%)$** | **$96\ (19.2\%)$** | — | — | $31.3\%$ reduction in specialist escalations; vulnerable to under-triage |
| **Policy S: Safeguarded DCRI Gateway** | **$193\ (38.6\%)$** | **$225\ (45.0\%)$** | **$82\ (16.4\%)$** | **$2\ (0.4\%)$** | **$94 / 96\ (97.9\%)$** | **$432 / 500\ (86.4\%)$** | High-uncertainty downgrades blocked; repeat exam mandated |

### Downward Reclassification & Safeguard Mechanics
1. **Unconstrained Monotonic Non-Inflationary Property (Policy A vs Policy B)**:
   Because $\text{DCRI}_{0.10} \le R_{\text{fusion}}$ for all valid packets, unconstrained uncertainty discounting never escalates a case:
   - **Total Reclassifications**: $96 / 500$ patients ($19.2\%$, $95\%$ Wilson CI: $[16.0\%, 22.9\%]$).
   - **Directionality**: Exactly $96 / 96$ ($100\%$) are downward shifts ($70$ from Tier 1 to Tier 0, $26$ from Tier 2 to Tier 1); $0$ upgrades ($0.0\%$).
   - **Escalation Reduction**: High-urgency specialist referrals drop from $83$ to $57$ (a **$31.3\%$ relative reduction**).
2. **Decision-Policy Safeguard Gateway (Policy S)**:
   Decouples the numerical research index ($\text{DCRI}_{0.10}$) from action-tier assignment by intercepting transitions when evidence is ambiguous:
   $$\text{Ambiguous}(\mathbf{x}) \iff \left(\sum_{i \in \mathcal{A}} U_i \ge 0.30\right) \lor \left(\min_{i \in \mathcal{A}} Q_i < 0.60\right)$$
   - **Selective Downgrade Suppression**: If ambiguous and $T_{\text{DCRI}} < T_{\text{base}}$, the case is held at its unpenalized fused risk tier ($T_{\text{final}} = T_{\text{base}}$), blocking **$94 / 96$ attempted downgrades ($97.9\%$)**.
   - **Escalation Preservation**: Urgent specialist escalations are preserved at $82 / 83$ ($98.8\%$ retention), preventing ambiguous or corrupted cases from slipping into routine care.
   - **Clear Evidence Downgrades**: Only $2$ cases ($0.4\%$) where cumulative uncertainty was strictly low ($U_{\text{sum}} < 0.30$) and input quality was high ($Q \ge 0.60$) are downgraded.

### Potential Clinical Consequences & Safety Nuance

> [!WARNING]
> **Hypothetical Policy Outcome — Not a Validated Clinical Decision**:
> 1. **The False-Downgrade Trade-Off**: In synthetic evaluations ($N=5,000$), unconstrained DCRI discounting reduces false specialist alarms but increases false downgrades of high-risk cases from $7.61\%$ to $12.72\%$ (nearly doubling in high-uncertainty regimes: $10.58\% \to 20.76\%$).
> 2. **Safeguard Resolution & Repeat-Exam Burden**: Policy S eliminates this under-escalation failure mode by refusing to lower triage urgency when uncertainty or sensor noise is high. However, this safety comes at a tangible operational cost: **$432 / 500$ encounters ($86.4\%$)** emit `FLAG_REPEAT_EXAM_AMBIGUOUS_EVIDENCE`. Safeguarding against false downgrades shifts the operational challenge from missed high-risk pathology to the management of repeat-diagnostic workloads.

*Artifact Traceability*:
- **Canonical Redesigned Cohort**: [`canonical_safeguarded_policy_results.json`](experiments/fusion/canonical_redesigned_rerun/canonical_safeguarded_policy_results.json) (Manifest SHA-256 `cf877d74e306...`); full metrics in [`04_canonical_c11_full_metrics.json`](experiments/fusion/canonical_redesigned_rerun/04_canonical_c11_full_metrics.json#L1601-L1658).
- **Baseline Decision Cohort**: [`safeguarded_policy_results.json`](experiments/fusion/dcri_policy_analysis/results/safeguarded_policy_results.json) (Manifest SHA-256 `1ff37fbeeba3...`); verified by [`verify_policy_artifacts.py`](verification/fusion/dcri_policy_analysis/verify_policy_artifacts.py).

---

## 7. Sensitivity Analysis and Ablation Studies

### A. One-Factor-At-A-Time (OFAT) Component Ablations ($N=500$)
Evaluates the isolated impact of deleting each component in ACARA-U on the canonical cohort:

| Ablation ID | Removed Component | $\overline{w}_R$ | $\overline{w}_F$ | $\overline{w}_C$ | Mean $\overline{R}$ | Mean $\overline{\text{DCRI}}$ | Reclassified vs A0 | Key Structural Finding |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **A0** | **Full B6 Reference** | **$42.71\%$** | **$22.31\%$** | **$34.98\%$** | **$0.261017$** | **$0.197624$** | **0** | Baseline reference configuration |
| **A1** | No Confidence ($\alpha=0$) | $43.01\%$ | $23.30\%$ | $33.69\%$ | $0.274761$ | $0.211368$ | $34\ (6.8\%)$ | Clinical authority drops $-1.3\%$; risk shifts $+0.0137$ |
| **A2** | No Reliability ($\beta=0$) | $40.26\%$ | $21.28\%$ | $38.46\%$ | $0.253192$ | $0.189799$ | $21\ (4.2\%)$ | Removes prior anchor; clinical authority increases $+3.5\%$ |
| **A3** | No Uncertainty ($\gamma=0$) | $36.42\%$ | $32.84\%$ | $30.74\%$ | $0.291355$ | $0.227962$ | $63\ (12.6\%)$ | Foot surges $+10.5\%$; highest risk distortion ($\Delta R = +0.0303$) |
| **A4** | No Quality ($\eta=0$) | $41.75\%$ | $22.34\%$ | $35.91\%$ | $0.259849$ | $0.196456$ | $2\ (0.4\%)$ | Minor authority shift on clean inputs ($\Delta w < 1.0\%$) |
| **A5** | No DCRI Discount ($\delta=0$) | $42.71\%$ | $22.31\%$ | $34.98\%$ | $0.261017$ | $0.261017$ | $96\ (19.2\%)$ | **Exact downstream orthogonality**: $\Delta w \equiv 0, \Delta R \equiv 0$ |

### B. Corrected Component Attribution Hierarchy
In the canonical execution path where clinical confidence reflects true decision-boundary distance ($C_C = 2\lvert p_1 - 0.5\rvert$, mean $0.776$):
1. **Uncertainty Penalty ($\gamma$)**: Primary active modulator. Removing $\gamma$ produces the largest risk deviation ($\Delta R = +0.030338$) and highest tier reclassifications ($63/500$).
2. **Confidence Weighting ($\alpha$)**: Second most influential factor. Removing $\alpha$ produces $\Delta R = +0.013744$ and $34/500$ reclassifications.
3. **Reliability Prior ($\beta$)**: Anchors authority to validated modalities ($\Delta R = -0.007825$, $21/500$ reclassifications).
4. **Quality Bonus ($\eta$)**: Gentle background term on clean data ($\Delta R = -0.001168$, $2/500$ reclassifications), activating decisively under sensor corruption.
5. **Downstream Discount Orthogonality ($\delta$)**: Setting $\delta = 0$ shifts DCRI by exactly $+0.063394$ without altering routing weights ($\Delta w \equiv 0.000000$).
*Artifact Traceability*: [`04_canonical_c11_full_metrics.json`](experiments/fusion/canonical_redesigned_rerun/04_canonical_c11_full_metrics.json#L1664-L1825); [Volume 16 Component Ablation Study](research/fusion/Volume_16_Component_Ablation_Study/README.md).

---

## 8. Input Degradation and Missing-Modality Safety

### Missing-Modality Safety Invariants across 8 Regimes ($N=500$)
Evaluates ACARA-U across all combinations of modality availability:

| Regime | Active Channels $\mathcal{A}$ | Mean Fused Risk $\overline{R}$ | Mean DCRI $\overline{\text{DCRI}}$ | Negative DCRI Count (%) | Zero Weight Leakage |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **RFC** | $\{R, F, C\}$ | $0.261017$ | $0.197624$ | $38\ (7.6\%)$ | Guaranteed ($w \equiv 0$ for $\notin \mathcal{A}$) |
| **RF** | $\{R, F\}$ | $0.297424$ | $0.252655$ | $18\ (3.6\%)$ | Guaranteed ($w_C \equiv 0.000000$) |
| **RC** | $\{R, C\}$ | $0.256940$ | $0.216416$ | $13\ (2.6\%)$ | Guaranteed ($w_F \equiv 0.000000$) |
| **FC** | $\{F, C\}$ | $0.316447$ | $0.276208$ | $7\ (1.4\%)$ | Guaranteed ($w_R \equiv 0.000000$) |
| **R** | $\{R\}$ | $0.255037$ | $0.254762$ | $15\ (3.0\%)$ | Singleton unity ($w_R \equiv 1.000000$) |
| **F** | $\{F\}$ | $0.521907$ | $0.403124$ | $6\ (1.2\%)$ | Singleton unity ($w_F \equiv 1.000000$) |
| **C** | $\{C\}$ | $0.113468$ | $0.105739$ | $4\ (0.8\%)$ | Singleton unity ($w_C \equiv 1.000000$) |
| **EMPTY** | $\emptyset$ | $0.000000$ | $0.000000$ | $0\ (0.0\%)$ | Fail-closed `NO_MODALITY_AVAILABLE` |

### Signal Degradation Monotonicity (D0 $\to$ D3)
Evaluated across 12 deterministic image and tabular degradation operators ($N=500$, seed 115):
- **Monotonic Authority Attenuation**: Real-time quality scores drop monotonically ($100.0\%$ packet monotonicity), yielding a **$36.7\%\text{--}51.7\%$ relative authority reduction** on corrupted channels.
- **Quality Term Isolation (B6 vs B5)**: Paired difference in authority attenuation under severe degradation ($D3$) was $D = -0.1309$ ($95\%$ bootstrap CI: $[-0.1319, -0.1300]$), confirming that $Q_i$ decisively down-weights degraded modalities beyond uncertainty alone.

*Artifact Traceability*: [`04_canonical_c11_full_metrics.json`](experiments/fusion/canonical_redesigned_rerun/04_canonical_c11_full_metrics.json#L344-L525); [Volume 08 Missing Modality](research/fusion/Volume_08_Missing_Modality_Robustness/README.md); [Volume 10 Input Degradation](research/fusion/Volume_10_Input_Degradation/README.md).

---

## 9. Synthetic Benchmark and Statistical Results

### A. Confirmatory Synthetic Oracle Benchmark ($N=5,000$, 10 Seeds)
A pre-registered synthetic oracle benchmark evaluated estimation error (MAE) against simulated ground-truth targets:

| Scenario / Condition | Sample Size ($N$) | Metric Evaluated | B6 vs B5 Difference | Empirical Scope & Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Overall Synthetic Benchmark** | $5,000$ | $\Delta_{\mathrm{MAE}}$ | $\mathbf{-0.001843}$ ($95\%\text{ CI: } [-0.002112, -0.001582]$) | Statistically significant ($p < 10^{-8}$); practical superiority threshold ($-0.005$) not met. |
| **Severe Degradation ($D3$)** | $385$ | $\Delta_{\mathrm{MAE}}$ | $\mathbf{-0.015416}$ ($95\%\text{ CI: } [-0.0170, -0.0139]$) | **$17.16\%$ relative MAE reduction** when sensor noise is severe. |
| **Clean Inputs ($D0$)** | $3,022$ | $\Delta_{\mathrm{MAE}}$ | $\mathbf{+0.000198}$ (Slightly favors B5) | Near-parity; quality term provides no benefit on clean data. |
| **Single-Modality Regimes** | $1,015$ | $\Delta_{\mathrm{MAE}}$ | $\mathbf{0.000000}$ (Exact equality) | Identical predictions due to singleton unity lock ($w \equiv 1.0$). |

### B. Confirmatory Multi-Cohort Tail Analysis ($S=30, N=15,000$)
To test whether ACARA-U improves sensitivity in sparse combination tails, 30 independent synthetic cohorts (seeds 401–430) were evaluated across distributions D1 (balanced), D2 (moderate tail), and D3 (strong tail):
- **Primary Estimand ($D3$ micro)**: $\Delta D_{\mathrm{tail}} = +0.000181$ with a $95\%$ hierarchical bootstrap confidence interval of $[-0.000024, +0.000377]$.
- **Confirmatory Finding**: Because the confidence interval crosses zero, the primary tail endpoint is **inconclusive**. Statistical equivalence is not established, and practical superiority of B6 over B5 in combination tails is unsupported.
*Artifact Traceability*: [Volume 05 Baseline Fusion](research/fusion/Volume_05_Baseline_Fusion/README.md); [Volume 09 Modality Combination Analysis](research/fusion/Volume_09_Modality_Combination_Analysis/README.md).

---

## 10. Reproducibility and Experiment Artifacts

### Canonical Manifest and Run Provenance
- **Canonical Execution Run ID**: `C11_CANONICAL_ADAPTER_RERUN_2026_10_13`

- **Canonical Packet Manifest**: `experiments/fusion/canonical_redesigned_rerun/01_canonical_redesigned_packet_manifest.json`

- **Manifest SHA-256**: `cf877d74e306d6dbc3ac7d9e2485b1c46da4cd64a8e35a82fe68274854dc4cd4`

- **Sample Count**: $N=500$ verified unique packets under deterministic seed 115.
- **Contract & Orchestration Test Suite**: 95/95 tests passed (`verification/fusion/contract/test_evidence_orchestration.py`).

### Verification Script and Artifact Registry

| Component / Experiment | Execution Script | Primary Output Artifact | Verification Status |
| :--- | :--- | :--- | :---: |
| **Evidence Orchestrator** | `src/fusion/adapters/orchestrator.py` | `01_canonical_redesigned_packet_manifest.json` | 95/95 Passed |
| **Canonical Independent Verification** | `verification/fusion/dcri_policy_analysis/test_canonical_safeguard.py` | `canonical_safeguarded_policy_results.json` | 5/5 Invariant Tests Passed |
| **Router Intermediate Benchmarks** | `experiments/fusion/run_canonical_rerun.py` | `03_router_intermediate_comparison.json` | Exact Reproducibility |
| **Full C11 Metrics Ladder** | `experiments/fusion/run_canonical_rerun.py` | `04_canonical_c11_full_metrics.json` | 8/8 Acceptance Gates Passed |
| **Baseline Decision Policy (Volume 14)** | `verification/fusion/dcri_policy_analysis/verify_policy_artifacts.py` | `dcri_policy_analysis/results/` | 12/12 Gates Passed (6 Artifacts) |
| **Policy Analysis Unit & Mutation Suite** | `verification/fusion/dcri_policy_analysis/` | Discover `test_*.py` | 39/39 Tests Passed |
| **Router Sanity & Monotonicity (Volume 15)**| `verification/fusion/test_router_sanity.py` | `Volume_15_Router_Sanity_Monotonicity/` | 16/16 Gates Passed |
| **Component Ablation Study (Volume 16)** | `experiments/fusion/ablation/` | `Volume_16_Component_Ablation_Study/` | 12/12 Gates Passed |
| **Decision-Level Evaluation (Volume 17)** | `verification/fusion/test_decision_eval.py` | `04_canonical_c11_full_metrics.json` | 16/16 Gates Passed |

### Installation and Environment Setup

```bash

# Clone the repository
git clone https://github.com/Dr-Venom29/FusionMedAI.git
cd FusionMedAI

# Create and activate virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

```

### Research Volume Index (Volumes 01–17)

| Volume | Title & Primary Investigation Focus | Verification & Artifact Status |
| :--- | :--- | :---: |
| [Volume 01](research/fusion/Volume_01_Research_Protocol/README.md) | Research Protocol Freeze & Routing Kernel Specification | Sealed Protocol |
| [Volume 02](research/fusion/Volume_02_Quality_Layer/README.md) | Unified Input Quality and Availability Layer | Verified Invariants |
| [Volume 03](research/fusion/Volume_03_Global_Reliability/README.md) | Global Modality Reliability Priors and Validation Evidence | Sealed Evidence |
| [Volume 04](research/fusion/Volume_04_ACARA_U_Router/README.md) | ACARA-U Dynamic Router & Behavioral Stress Benchmarking | Simplex Invariants |
| [Volume 05](research/fusion/Volume_05_Baseline_Fusion/README.md) | Multimodal Baseline Ladder (B1–B6) & Comparative Evaluation | Baselines Evaluated |
| [Volume 06](research/fusion/Volume_06_DCRI_Aggregation/README.md) | DCRI Risk Aggregation & Uncertainty Discounting | Bounds Certified |
| [Volume 07](research/fusion/Volume_07_Conflict_Analysis/README.md) | Cross-Modality Conflict & Discordance Analysis | Orthogonality Confirmed |
| [Volume 08](research/fusion/Volume_08_Missing_Modality_Robustness/README.md) | Missing Modality Robustness & Authority Redistribution | Fail-Closed Verified |
| [Volume 09](research/fusion/Volume_09_Modality_Combination_Analysis/README.md) | Modality-Combination Distribution & Confirmatory Tail Analysis | Multi-Cohort Sealed |
| [Volume 10](research/fusion/Volume_10_Input_Degradation/README.md) | Input Degradation Response & Quality Isolation | 12 Operators Audited |
| [Volume 11](research/fusion/Volume_11_Fusion_Calibration/README.md) | Modality Calibration Impact on Decision Fusion | Authority Rebalanced |
| [Volume 12](research/fusion/Volume_12_ACARA_U_Parameter_Sensitivity/README.md) | ACARA-U Parameter & Weighting Sensitivity Analysis | 23 Sweeps Bounded |
| [Volume 13](research/fusion/Volume_13_DCRI_Delta_Selection/README.md) | DCRI Global Uncertainty Penalty Selection ($\delta^* = 0.10$) | Selection Certified |
| [Volume 14](research/fusion/Volume_14_DCRI_Decision_Policy_Analysis/README.md) | DCRI Decision Policy & Operating Behavior Analysis | 3-Tier Policy Mapped |
| [Volume 15](research/fusion/Volume_15_Router_Sanity_Monotonicity/README.md) | ACARA-U Residual Router Sanity & Monotonicity Analysis | Monotonicity Verified |
| [Volume 16](research/fusion/Volume_16_Component_Ablation_Study/README.md) | ACARA-U Component Ablation Study & Factor Deletions | Orthogonality Proven |
| [Volume 17](research/fusion/Volume_17_Decision_Level_Evaluation/README.md) | Final Decision-Level Evaluation of ACARA-U Framework | 5-Dimension Verified (Algorithmic) |

> [!NOTE]
> **5-Dimension Algorithmic Verification Scope (Volume 17)**:
> The "5-Dimension Verified" status refers strictly to passing all 16/16 computational verification gates and 46/46 automated unit, invariant, and mutation tests across the five evaluated operational dimensions on standardized decision packets:
> 1. **Baseline Ladders (B1–B6)**: Exact mathematical reproduction of risk and DCRI distributions.
> 2. **Availability Regimes (8 permutations)**: Simplex conservation ($\sum w_i = 1.0$), zero leakage to inactive channels ($w_i = 0$), and fail-closed termination.
> 3. **Degradation Severities (D0–D3)**: Monotonic quality attenuation and non-inflationary risk behavior under sensor corruption.
> 4. **Conflict Stratification (LOW/MOD/HIGH)**: Continuous discordance quantification and correlation decoupling from epistemic uncertainty.
> 5. **Calibration Conditions**: Softened authority reallocation with bounded entropy preservation.

> 
> *Passing computational software checks and invariant gates establishes pipeline reproducibility and numerical consistency; it is not equivalent to, and does not establish, clinical validity or prospective clinical safety.*

---

## 11. Clinical Limitations and Intended Use
1. **Absence of Paired Real-World Validation**: Public medical datasets do not provide simultaneous retinal fundus scans, foot-ulcer photographs, and hospital EHR records for the same individuals. All multimodal fusion mechanics are evaluated on controlled decision packets and synthetic degradation models. Prospective clinical safety, multi-center generalization, and patient outcomes remain unestablished.
2. **Rejection of Universal Clinical Target**: The three underlying models predict distinct biological and operational endpoints (retinopathy progression, wound necrosis, and acute hospital readmission). The fused scalar $R_{\text{fusion}}$ and index $\text{DCRI}$ represent **review-prioritization and decision-support triage scores**, not a unified biological complication probability. In clinical settings, discrete modality predictions must be reviewed in their native clinical context.
3. **Synthetic Oracle Practical Superiority Boundary**: While quality-aware routing (B6) achieves a statistically significant error reduction under synthetic oracle evaluation ($\Delta_{\mathrm{MAE}} = -0.001843, p < 10^{-8}$), it does not meet the prespecified practical superiority threshold of $-0.005$. Furthermore, on clean uncorrupted data, B5 slightly outperforms B6 ($\Delta_{\mathrm{MAE}} = +0.000198$). Quality terms should only be incorporated when sensor corruption is a documented operational hazard.
4. **Under-Escalation Risk and Policy S Safeguard Resolution**: Although the unconstrained uncertainty penalty ($\delta^* = 0.10$) eliminates upward risk inflation and reduces specialist escalation burden by $31.3\%$, it increases the false-downgrade rate on synthetic oracle high-risk packets from $7.61\%$ to $12.72\%$. To resolve this without abandoning uncertainty discounting, the framework introduces the **Decision-Policy Safeguard Gateway (Policy S)**. Policy S preserves $\text{DCRI}_{0.10}$ as the research index while suppressing downgrades whenever evidence is ambiguous ($U_{\text{sum}} \ge 0.30 \lor \min(Q) < 0.60$), blocking $94/96$ ($97.9\%$) of unconstrained downgrades and preserving $82/83$ specialist escalations. The operational trade-off is a high repeat-exam burden ($86.4\%$ flag rate), ensuring uncertain cases receive diagnostic re-testing rather than premature discharge.
5. **Computational Verification vs Clinical Regulatory Approval**: Passing 95/95 orchestration unit tests and 16/16 decision-level invariant gates confirms internal algorithmic consistency and reproducibility within this codebase. It does not constitute medical device certification, software-as-a-medical-device (SaMD) clearance, or evidence of clinical efficacy.

---

## Contributors

| [<img src="https://github.com/Dr-Venom29.png" width="100px;" alt="Dr-Venom29"/><br /><sub><b>Dr-Venom29</b></sub>](https://github.com/Dr-Venom29) | [<img src="https://github.com/NoBodyKnows3000.png" width="100px;" alt="NoBodyKnows3000"/><br /><sub><b>NoBodyKnows3000</b></sub>](https://github.com/NoBodyKnows3000) | [<img src="https://github.com/Chandu45678.png" width="100px;" alt="Chandu45678"/><br /><sub><b>Chandu45678</b></sub>](https://github.com/Chandu45678) | [<img src="https://github.com/NithinVN.png" width="100px;" alt="NithinVN"/><br /><sub><b>NithinVN</b></sub>](https://github.com/NithinVN) |
| :---: | :---: | :---: | :---: |

---

## License
This project is distributed under the MIT License. See [LICENSE](LICENSE) for details.
