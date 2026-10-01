# Clinical Modality Research Documentation

Welcome to the research documentation suite for the **Clinical Tabular Modality** of **FusionMedAI**.

This modality focuses on predicting 30-day all-cause hospital readmission for diabetic patients using structured electronic health records (EHR) under frozen patient-level partitions and rigorous empirical verification.

---

## Research Volumes

| Volume | Phase | Title | Status | Artifacts & Documentation |
|:---:|:---:|:---|:---:|:---|
| **05** | **C5** | [Tabular Architecture Benchmarking & Empirical Optimization](Volume_05_Architecture_Benchmarking/README.md) | **Completed** | 7 architectures, Bayesian HPO, subgroup auditing, complexity profiling |
| **06** | **C6** | [Model Explainability & TreeSHAP Attribution Analysis](Volume_06_Explainability/README.md) | **Completed** | Global/local TreeSHAP, directionality, clinical taxonomy, error analysis |
| **07** | **C7** | Probability Calibration & Reliability Assessment | *Planned* | Post-hoc scaling (Platt, Isotonic, Temperature, Vector) |
| **08** | **C8** | Prediction Uncertainty Estimation | *Planned* | Epistemic/aleatoric uncertainty profiling |
| **09** | **C9** | Robustness & Subgroup Auditing | *Planned* | Intersectional fairness & distribution shift |
| **10** | **C10** | External Clinical Validation | *Planned* | Independent EHR cohort validation |
| **11** | **C11** | Clinical Module Integration | *Planned* | Production interface & multimodal contract |

---

## Core Invariants & Governance

1. **Frozen Partitioning**: Canonical patient-level split ($N_{\text{train}}=69,519$, $N_{\text{val}}=14,911$, $N_{\text{test}}=14,913$) with zero cross-split patient contamination.
2. **Representation Contract**: Locked $119$-dimensional feature space generated strictly by `ClinicalPreprocessor`.
3. **Evaluation Protocol**: Zero test-label snooping during hyperparameter optimization, threshold selection, and explainability attribution generation.
4. **Primary Selected Candidate**: Frozen **CatBoost HPO** (`depth=4`, `learning_rate=0.1383`, `iterations=350`, `l2_leaf_reg=2.911`, `subsample=0.655`, `random_seed=42`).
