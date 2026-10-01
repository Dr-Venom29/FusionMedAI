# Document 03: Bootstrap Ensemble Training & Member Dynamics

## 1. Bootstrap Resampling Architecture

The primary uncertainty estimation backbone consists of $M = 50$ distinct CatBoost models fitted on independent uniform bootstrap draws of the canonical training split ($N_{\text{train}} = 69,519$).

Each bootstrap resample contains on average:
$$1 - \frac{1}{e} \approx 63.2\% \quad (43,943\text{ unique encounters})$$
with $36.8\%$ ($25,576\text{ encounters}$) represented via multiple duplicate draws.

---

## 2. Ensemble Specification & Training Profiling

| Property | Value | Protocol Status |
| :--- | :---: | :---: |
| **Ensemble Size ($M$)** | $50\text{ models}$ | Selected by empirical convergence audit |
| **Tree Depth** | $4$ | Strictly locked from Phase C5 |
| **Learning Rate ($\eta$)** | $0.1383$ | Strictly locked from Phase C5 |
| **Tree Count per Member** | $350\text{ oblivious trees}$ | Total ensemble capacity = $17,500\text{ trees}$ |
| **$L_2$ Leaf Regularization** | $2.911$ | Strictly locked from Phase C5 |
| **Subsampling Ratio** | $0.655$ | Applied per tree split |
| **Feature Dimensions** | $D = 119$ | Locked Preprocessor representation |
| **Seed Sequence** | $S_m \in \{43, 44, \dots, 92\}$ | Deterministic reproducibility |

---

## 3. Member Prediction Diversity

To verify that bootstrap models capture non-trivial parameter variation rather than producing degenerate duplicate outputs, member diversity was analyzed across test encounters:

```
Inter-Model Prediction Correlation: r̄ = 0.9412  (Range: [0.9105, 0.9688])
Average Member-Wise Prediction Dispersion: σ̄ = 0.0219
Maximum Single-Encounter Standard Deviation: σ_max = 0.1492
```

The high correlation ($r \approx 0.94$) confirms that all ensemble members capture the primary underlying risk surface, while the residual dispersion ($\sigma_p \approx 0.022$) reflects genuine parameter estimation variability in sparse clinical feature regions.
