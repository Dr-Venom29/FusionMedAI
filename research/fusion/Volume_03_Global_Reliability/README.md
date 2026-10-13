# Volume 03 — Global Modality Reliability (Phase C11.3)

## Volume Overview & Scope

This volume formalizes and documents the computation and freezing of the empirical **Global Modality Reliability** prior ($R_i \in [0.0, 1.0]$) for each modality in **FusionMedAI**:

$$
R_i = \frac{1}{2} \cdot \left[ \text{AUC}_i + (1 - \text{ECE}_i) \right]
$$

Reliability represents an immutable, modality-level prior grounded strictly in frozen validation split evidence. It balances historical discriminative capacity (50%) and probability calibration fidelity (50%) before dynamic multimodal routing begins in ACARA-U (Phase C11.4).

---

## Documents in this Volume

| Document | Focus Area | Key Formalizations |
| :--- | :--- | :--- |
| [`01_Reliability_Protocol.md`](./01_Reliability_Protocol.md) | Methodological Governance | Formula lock, zero test-set peeking, distinction between $R_i$, $Q_i$, $C_i$, and $U_i$. |
| [`02_Validation_Evidence.md`](./02_Validation_Evidence.md) | Empirical Validation Data | Validation split sizes ($N_{\text{val}}=366, 1006, 14911$), frozen checkpoints, and logits. |
| [`03_AUC_and_ECE.md`](./03_AUC_and_ECE.md) | Metric Formalization | Macro OvR AUC ($K=5$ Retina, $K=4$ Foot), binary AUC (Clinical), 10-bin equal-frequency ECE. |
| [`04_Global_Reliability.md`](./04_Global_Reliability.md) | Exact Computed Coefficients | Empirical values: $R_R=0.929956, R_F=0.922266, R_C=0.825382$. |
| [`05_Freeze_Report.md`](./05_Freeze_Report.md) | Freeze Report & Router Handoff | Provenance manifest, JSON artifacts, and downstream handoff to ACARA-U router. |

---

## Summary of Frozen Reliability Priors

| Modality ($i$) | Frozen Validation Backbone | Validation AUC ($\text{AUC}_i$) | Validation ECE ($\text{ECE}_i$) | Calibration Score ($1-\text{ECE}_i$) | Global Reliability ($R_i$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Retina ($R$)** | EfficientNet-B3 + Temperature Scaling ($N_{\text{val}}=366$) | **0.911149** | **0.051237** | **0.948763** | **0.929956** |
| **Foot ($F$)** | EfficientNet-B3 + Vector Scaling ($N_{\text{val}}=1006$) | **0.884435** | **0.039903** | **0.960097** | **0.922266** |
| **Clinical ($C$)** | CatBoost HPO + Isotonic Calibration ($N_{\text{val}}=14911$) | **0.650764** | **0.000000** | **1.000000** | **0.825382** |

*Note: All quantities are frozen constants derived exclusively from validation data and passed through `RetinaAdapter`, `FootAdapter`, and `ClinicalAdapter` into `EvidenceOrchestrator` and `ACARAUv2Router`.*

### Validation vs. Locked Test Calibration Split Provenance (Clinical Modality)
- **CatBoost HPO Training Split**: `datasets/clinical/processed/splits/train.csv` ($N_{\text{train}} = 70,118$ rows / $69,519$ preprocessed encounters).
- **Isotonic Calibrator Fitting & Validation Evaluation Split**: `datasets/clinical/processed/splits/val.csv` ($N_{\text{val}} = 14,911$ encounters). Because non-parametric Isotonic Regression is fitted and evaluated on `val.csv`, $\text{ECE}_{\text{val}} = 0.000000$ ($2.48 \times 10^{-15}$) is an in-sample validation calibration measurement, yielding $R_C = \frac{0.650764 + (1 - 0.000000)}{2} = 0.825382$. Evaluating 5-fold stratified cross-fitting of Isotonic Regression strictly inside `val.csv` yields out-of-fold $\text{ECE}_{\text{val,5-fold OOF}} = 0.006229$ ($R_{C,\text{OOF}} = 0.822268$, $\Delta R_C = -0.003114$).
- **Locked Test Split**: `datasets/clinical/processed/splits/test.csv` ($N_{\text{test}} = 14,913$ encounters, strictly isolated from router parameterization) exhibits raw test $\text{ECE}_{\text{test}} = 0.003198$ and isotonic test $\text{ECE}_{\text{test}} = 0.006179$.

