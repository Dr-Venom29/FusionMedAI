# Methodological Boundaries & Transparent Scope

## 1. Parameter Sensitivity vs Hyperparameter Optimization

A critical scientific distinction separates Phase C11.12 from empirical hyperparameter tuning:

> **Phase C11.12 is a parameter sensitivity characterization, NOT a coefficient optimization study.**

### Explicit Boundary Rules:
1. **No Target-Based Coefficient Selection**: Coefficients were not evaluated against an empirical performance target, loss function, or clinical objective to pick a "winning" configuration.
2. **No Post-Hoc Re-Tuning**: Finding that $\beta=1.75$ or $\alpha=1.25$ slightly increases an index does not justify altering the reference configuration. $\Theta_0 = (1.0, 1.5, 1.0, 0.5)$ remains frozen.
3. **Prevention of Test-Set Leakage**: The $N=500$ controlled packet cohort is a diagnostic evaluation benchmark; tuning hyperparameter weights on this cohort would constitute retrospective overfitting.

---

## 2. Unpaired Dataset Boundary & Decision Index Integrity

1. **Unpaired Cohort Architecture**: The underlying retinal (Messidor-2/APTOS), foot (DFUC2020), and clinical tabular EHR (MIMIC-IV) datasets remain unpaired. No synthetic multimodal patient ground truth is constructed.
2. **Decision Indices, Not Clinical Probabilities**:
   - $R_{\text{fusion}} = \sum_{i \in \mathcal{A}} w_i r_i$ is a decision-level weighted risk index.
   - $\text{DCRI} = R_{\text{fusion}} - \delta \sum_{i \in \mathcal{A}} U_i$ is a composite decision-confidence index.
   - Neither metric is treated as a clinically calibrated event probability.
3. **Provisional $\delta = 0.20$ Scope**: The uncertainty penalty factor $\delta = 0.20$ is utilized strictly as the frozen provisional baseline from Phase C11.6; formal optimization and trade-off characterization of $\delta$ is reserved for Phase C11.13.

---

## 3. Stability Classification Taxonomy

Phase C11.12 classifies hyperparameter perturbations into three rigorous scientific regimes:

- 🟢 **Stable**: Tested parameter perturbations produce bounded authority adjustments without invariant violations, routing collapse, or numerical instability.
- 🟡 **Sensitive**: Perturbations produce substantial routing redistribution while preserving mathematical invariants and the expected directional response of the perturbed coefficient.
- 🔴 **Unstable**: Perturbations trigger routing collapse ($H \to 0$), numerical overflow/underflow, invariant violations, or clearly inconsistent/non-monotonic responses under the tested perturbation design.

**Empirical Conclusion**: All 23 named evaluations (19 unique parameter vectors) in the tested perturbation domain fall strictly within the 🟢 **Stable** and 🟡 **Sensitive** categories; **zero unstable regimes were observed within the tested parameter domain**.
