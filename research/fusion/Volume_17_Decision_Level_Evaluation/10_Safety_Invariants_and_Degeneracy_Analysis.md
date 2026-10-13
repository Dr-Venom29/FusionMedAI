# Chapter 10: Safety Invariants and Degeneracy Analysis

## 1. Safety Invariant Definitions
A robust clinical decision engine must satisfy rigorous mathematical invariants to prevent invalid outputs, numerical instability, or unauthorized risk propagation. Phase C11.17 independently verifies five primary mathematical and architectural invariants:

1. **Probability Simplex Invariant (Non-Empty Regimes)**:

   $$\sum_{i \in \mathcal{A}} w_i = 1.0 \quad \text{and} \quad 0.0 \le w_i \le 1.0 \quad \forall i \in \mathcal{A}$$

   *Scope*: Applies strictly to non-empty availability regimes ($\lvert\mathcal{A}\rvert \ge 1$). The empty regime ($\mathcal{A} = \emptyset$) is governed by the dedicated fail-closed contract below.
2. **Hard Availability Mask Invariant**:

   $$w_k \equiv 0.000000 \quad \forall k \notin \mathcal{A}$$

   Ensures that unavailable sensors receive strictly zero routing weight and do not contribute to fused risk or uncertainty sums.
3. **Finite Score Invariant**:

   $$R_{\mathrm{fusion}}, \mathrm{DCRI}, P_U \in \mathbb{R} \quad (\text{strictly finite numbers: no NaN, no }+\infty\text{, no }-\infty)$$

4. **Fail-Closed EMPTY Degeneracy Contract**:

   $$\mathcal{A} = \emptyset \implies w_i = 0.0 \quad \forall i, \quad R_{\mathrm{fusion}} = 0.0, \quad P_U = 0.0, \quad \mathrm{DCRI} = 0.0, \quad \text{status} = \mathtt{NO\_MODALITY\_AVAILABLE}$$

5. **Singleton Unity Invariance**:

   $$\lvert\mathcal{A}\rvert = 1 \implies w_{\mathrm{active}} \equiv 1.000000, \quad w_{\mathrm{inactive}} \equiv 0.000000$$

## 2. Invariant Verification Results across Tested Populations
Every invariant was audited by the independent verification suite (`verify_decision_evaluation_artifacts.py`). The exact populations, checked denominators, and verification results are detailed below:

| Safety Invariant | Target Population & Scope | Records Checked | Observed Violations | Verification Outcome |
| :--- | :--- | :---: | :---: | :---: |
| **Probability Simplex Constraint** | Baseline cohorts (B1–B6, 6 policies $\times$ 500 packets) | 3,000 | 0 | **PASSED** (Gate C17-03) |
| **Hard Availability Masking** | 8 availability regimes across 4 policies (B1, B2, B5, B6 $\times$ 500 packets) | 16,000 | 0 | **PASSED** (Gate C17-04) |
| **Fail-Closed EMPTY Contract** | Empty regime across 4 evaluated policies (B1, B2, B5, B6 $\times$ 500 packets) | 2,000 | 0 | **PASSED** (Gate C17-05) |
| **Singleton Unity Lock** | Unimodal singletons (R, F, C across B1, B2, B5, B6 $\times$ 500 packets) | 6,000 | 0 | **PASSED** (Gate C17-06) |
| **Exact DCRI Formula & Finiteness** | Entire canonical record volume across all evaluation conditions | 24,000 | 0 | **PASSED** (Gate C17-07) |

*Definition of "PASSED"*: Indicates that zero mathematical violations or invalid numerical records were encountered across the declared test cohorts under the automated verifier gates. It verifies software contract compliance within the tested experimental parameter domain; it does not demonstrate the absence of all possible runtime anomalies, software defects, or clinical risks in uncurated deployment settings.

## 3. Degeneracy Invariance & Numerical Analysis

### 3.1 Unimodal Singleton Stability
When exactly one modality channel is available ($\lvert\mathcal{A}\rvert = 1$ with active channel $i$):

$$\tilde{z}_i = z_i, \quad \tilde{z}_{j \neq i} = -\infty \implies w_i = \frac{\exp(z_i)}{\exp(z_i) + \sum_{j \neq i} 0} \equiv 1.000000$$

All singleton outputs evaluate to unimodal predictions without division-by-zero or numerical drift across all 6,000 evaluated singleton records.

### 3.2 Softmax Numerical Stability
Before exponential evaluation in the softmax router, the engine validates that input logits are strictly finite and applies maximum subtraction:

$$\tilde{z}_i = z_i - \max_{j \in \mathcal{A}} z_j$$

This guarantees that $\exp(\tilde{z}_i) \in (0, 1]$, preventing floating-point overflow for finite logits within the tested hyperparameter bounds ($\alpha = 1.0, \beta = 1.5, \gamma = 1.0, \eta = 0.5$). If upstream inputs contain non-finite numbers (NaN or $\pm\infty$), `validate_record()` and the verifier reject the encounter before softmax computation.
