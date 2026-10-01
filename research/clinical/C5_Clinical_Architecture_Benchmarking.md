# Clinical Phase C5: Tabular Architecture Benchmarking

## Executive Summary
This document provides the high-level research summary for **Phase C5 Tabular Architecture Benchmarking** of the Clinical Modality in FusionMedAI. Complete, detailed documentation across all 12 research sections is available in [`research/clinical/Volume_05_Architecture_Benchmarking/`](Volume_05_Architecture_Benchmarking/README.md).

---

## 1. Experimental Setup & Representation Contract
- **Dataset Partition**: Frozen Phase C2 patient-level partitioned splits ($N_{\text{train}}=69,519$, $N_{\text{val}}=14,911$, $N_{\text{test}}=14,913$) with binary 30-day all-cause readmission endpoint ($11.16\%$ prevalence).
- **Representation Contract**: Locked $119$-dimensional feature space produced strictly by `ClinicalPreprocessor` fitted on the training split.
- **Optimization Target**: Precision-Recall Area Under the Curve (**PR-AUC**) on the validation partition, preventing data leakage into the locked test split.

---

## 2. 7-Model Benchmarking Results (Locked Test Set: $N=14,913$)

| Architecture | Model Family | Test ROC-AUC | Test PR-AUC | Test Brier Score | Test ECE (10 Bins) | Test Sensitivity ($\theta=0.20$) | Test Specificity ($\theta=0.20$) | Inference Latency (ms/1k) | Artifact Size (KB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CatBoost (Tuned HPO)** | GBDT (Oblivious) | $\mathbf{0.6504}$ | $\mathbf{0.2063}$ | $\mathbf{0.0952}$ | $0.0053$ | $\mathbf{17.01\%}$ | $93.61\%$ | $2.17\text{ ms}$ | $\mathbf{184.2\text{ KB}}$ |
| **CatBoost (Default)** | GBDT (Oblivious) | $0.6472$ | $0.2038$ | $0.0953$ | $0.0066$ | $16.47\%$ | $94.06\%$ | $2.31\text{ ms}$ | $415.6\text{ KB}$ |
| **XGBoost** | GBDT (Greedy) | $0.6467$ | $0.2035$ | $0.0953$ | $0.0051$ | $15.92\%$ | $94.18\%$ | $\mathbf{1.89\text{ ms}}$ | $260.2\text{ KB}$ |
| **LightGBM** | GBDT (Histogram) | $0.6461$ | $0.2038$ | $0.0953$ | $\mathbf{0.0045}$ | $15.80\%$ | $94.40\%$ | $3.19\text{ ms}$ | $318.2\text{ KB}$ |
| **Logistic Regression (L2)** | Linear (L-BFGS) | $0.6446$ | $0.1969$ | $0.0958$ | $0.0080$ | $14.41\%$ | $95.06\%$ | $1.13\text{ ms}$ | $\mathbf{1.8\text{ KB}}$ |
| **Logistic Regression (EN)** | Linear (SAGA) | $0.6445$ | $0.1971$ | $0.0958$ | $0.0084$ | $14.35\%$ | $95.07\%$ | $\mathbf{0.74\text{ ms}}$ | $\mathbf{1.8\text{ KB}}$ |
| **Random Forest** | Bagged Trees | $0.6422$ | $0.1991$ | $0.0959$ | $0.0098$ | $9.71\%$ | $\mathbf{97.31\%}$ | $44.83\text{ ms}$ | $5,519.4\text{ KB}$ |
| **TabNet** | Neural Attention | $0.6252$ | $0.1887$ | $0.0962$ | $0.0105$ | $15.32\%$ | $93.59\%$ | $19.40\text{ ms}$ | $1,099.7\text{ KB}$ |

---

## 3. Key Findings

1. **Discrimination Leadership**: **CatBoost Tuned** achieved the highest test ROC-AUC ($0.6504$) and test PR-AUC ($0.2063$).
2. **Calibration Leadership**: **LightGBM** achieved the lowest test ECE ($0.0045$) and fastest training time ($0.58\text{ s}$).
3. **Inference Latency Leadership**: **XGBoost** produced the lowest measured inference latency among boosted trees ($1.89\text{ ms} / 1\text{k}$).
4. **Primary C6 Fusion Candidate**: **CatBoost Tuned** is designated as the primary tabular candidate for multimodal fusion in Phase C6. Clinical deployment remains outside the scope of Phase C5 and requires additional prospective validation.

---

## 4. Volume 05 Navigation

For in-depth analysis, refer to individual documents in [`Volume_05_Architecture_Benchmarking/`](Volume_05_Architecture_Benchmarking/README.md):
- [01_modeling_contract.md](Volume_05_Architecture_Benchmarking/01_modeling_contract.md) — Protocol & representation invariants
- [02_catboost.md](Volume_05_Architecture_Benchmarking/02_catboost.md) — CatBoost architecture & oblivious trees
- [03_tabnet.md](Volume_05_Architecture_Benchmarking/03_tabnet.md) — TabNet sequential sparse attention
- [04_hyperparameter_optimization.md](Volume_05_Architecture_Benchmarking/04_hyperparameter_optimization.md) — Bayesian TPE HPO protocol & trial results
- [05_architecture_comparison.md](Volume_05_Architecture_Benchmarking/05_architecture_comparison.md) — 7-model performance comparison
- [06_calibration_analysis.md](Volume_05_Architecture_Benchmarking/06_calibration_analysis.md) — Probability calibration & reliability
- [07_threshold_analysis.md](Volume_05_Architecture_Benchmarking/07_threshold_analysis.md) — Clinical decision thresholds ($\theta \in [0.10, 0.50]$)
- [08_subgroup_analysis.md](Volume_05_Architecture_Benchmarking/08_subgroup_analysis.md) — Demographic & clinical subgroup fairness audit
- [09_error_analysis.md](Volume_05_Architecture_Benchmarking/09_error_analysis.md) — Classification error breakdown
- [10_complexity_analysis.md](Volume_05_Architecture_Benchmarking/10_complexity_analysis.md) — Computational complexity & latency profiling
- [11_conclusion.md](Volume_05_Architecture_Benchmarking/11_conclusion.md) — Architectural conclusions & selection
