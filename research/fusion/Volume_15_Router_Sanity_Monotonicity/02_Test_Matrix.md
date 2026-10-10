# Chapter 02: Test Matrix & Evaluation Methodology

## 1. Modality Availability Regimes

The evaluation spans all 7 valid active regimes plus the empty degenerate regime:

| Regime Symbol | Modality Composition | Modality Count $\lvert \mathcal{A} \rvert$ | Active Channels |
| :---: | :--- | :---: | :--- |
| **R** | Retina Only | 1 | `retina` |
| **F** | Foot Only | 1 | `foot` |
| **C** | Clinical Only | 1 | `clinical` |
| **RF** | Retina + Foot | 2 | `retina`, `foot` |
| **RC** | Retina + Clinical | 2 | `retina`, `clinical` |
| **FC** | Foot + Clinical | 2 | `foot`, `clinical` |
| **RFC** | Retina + Foot + Clinical | 3 | `retina`, `foot`, `clinical` |
| **EMPTY** | No Modalities Available | 0 | None (Fail-Closed) |

---

## 2. Isolated Single-Input Monotonicity Test Matrix

For each of the 4 channel attributes ($C_i, R_i, U_i, Q_i$), perturbations are applied across the interior and boundary grid while holding all other channel attributes and modalities strictly constant.

### 2.1 Grid Configuration
- **Base Grid Values:** $x \in \{0.2, 0.4, 0.6, 0.8\}$
- **Perturbation Steps:** $\Delta x \in \{+0.05, +0.10, +0.20\}$
- **Trial Count Calculation:**

  $$
  \text{Total Trials per Attribute} = \sum_{\text{regime} \in \text{ACTIVE}} \lvert \mathcal{A}_{\text{regime}} \rvert \times \lvert \text{Base Grid} \rvert \times \lvert \text{Steps} \rvert
  $$

  $$
  = (1 + 1 + 1 + 2 + 2 + 2 + 3) \times 4 \times 3 = 12 \times 12 = 144 \text{ trials}
  $$

### 2.2 Directionality Expectations by Cardinality

| Attribute Tested | Multi-Modality Active Set ($\lvert \mathcal{A} \rvert \ge 2$) | Single-Modality Set ($\lvert \mathcal{A} \rvert = 1$) | Tolerance Threshold |
| :--- | :--- | :--- | :--- |
| **Confidence ($C_i$)** | $\Delta w_i > \epsilon$ (Strict Increase) | $\Delta w_i = 0.0, \, w_i = 1.0$ (Invariant) | $\epsilon = 10^{-7}$ |
| **Reliability ($R_i$)** | $\Delta w_i > \epsilon$ (Strict Increase) | $\Delta w_i = 0.0, \, w_i = 1.0$ (Invariant) | $\epsilon = 10^{-7}$ |
| **Uncertainty ($U_i$)** | $\Delta w_i < -\epsilon$ (Strict Decrease) | $\Delta w_i = 0.0, \, w_i = 1.0$ (Invariant) | $\epsilon = 10^{-7}$ |
| **Quality ($Q_i$)** | $\Delta w_i > \epsilon$ (Strict Increase) | $\Delta w_i = 0.0, \, w_i = 1.0$ (Invariant) | $\epsilon = 10^{-7}$ |

---

## 3. Mathematical Invariants Test Matrix

| Invariant Check | Test Description | Sample Size / Grid | Passing Criterion |
| :--- | :--- | :--- | :--- |
| **Simplex Conservation** | Verifies $\sum_{i \in \mathcal{A}} w_i = 1.0$ and $w_i \ge 0$. | 100 random samples per regime (700 total) | $\lvert \sum w_i - 1.0 \rvert < 10^{-10}$, $w_i \ge 0.0$ |
| **Hard Availability Masking** | Verifies $A_j = 0 \implies w_j = 0.0$ exact. | 50 random samples per partial regime (300 total) | Zero weight leakage ($w_j = 0.0$) |
| **Reference Softmax Shift Invariance** | Evaluates $\text{softmax}(z + c) \equiv \text{softmax}(z)$ in reference kernel. | 11 shifts $c \in \{-500, -100, -50, -10, -1, 0, 1, 10, 50, 100, 500\}$ | $\max \lvert \Delta w_i \rvert < 10^{-12}$ |
| **Production Route Shift Experiment** | Evaluates uniform $+0.10$ confidence shift across active channels in `router.route()`. | Dual route invocation on baseline vs uniformly shifted input | $\max \lvert \Delta w_i \rvert < 10^{-12}$ |
| **Reference Kernel Permutation Invariance** | Evaluates order independence across all channel permutations in normalization kernel. | All $3! = 6$ channel permutations | $\max \lvert \Delta w_i \rvert = 0.0$ |
| **Numerical Stability** | Evaluates 5 boundary inputs in `route()` and extreme differentials in kernel. | 5 edge scenarios + extreme logit differentials | All outputs finite, sum $= 1.0 \pm 10^{-10}$ |
| **Determinism** | Verifies bitwise reproducibility. | 25 consecutive executions on identical input | Exact bitwise weight equality |

---

## 4. Edge Cases & Fault Injection Matrix

| Test Category | Injected Condition | Expected Response | Target Gate |
| :--- | :--- | :--- | :--- |
| **Zero Modalities** | $A_{\text{retina}} = A_{\text{foot}} = A_{\text{clinical}} = 0$ | Status: `NO_MODALITY_AVAILABLE`, $w = (0, 0, 0)$ | S15-11 |
| **Masked Value Corruption** | Altering $C_j, U_j$ on inactive channel $j$ | Active weights $w_i$ remain bitwise invariant | S15-12 |
| **Input Validation** | Passing $\text{NaN}$, $\infty$, out-of-bounds scalars, quality-on-inactive, or reliability mismatches | Raises concrete `RouterContractValidationError` (`ValueError` subclass); invalid coefficients raise `RouterCoefficientError` (`ValueError` subclass) | S15-10 |
| **Mutation 1** | Inverted uncertainty sign ($+\gamma U_i$) in router kernel | Evaluator fails uncertainty monotonicity ($\Delta w_i > 0$); verifier rejects at S15-04 | S15-15 |
| **Mutation 2** | Inverted confidence sign ($-\alpha C_i$) in router kernel | Evaluator fails confidence monotonicity ($\Delta w_i < 0$); verifier rejects at S15-02 | S15-15 |
| **Mutation 3** | Inactive channel weight leakage ($w_j > 0$) in `test_results.json` | Verifier detects hard availability masking violation; rejects at S15-07 | S15-15 |
| **Mutation 4** | Corrupted SHA-256 hash in `freeze_manifest.json` | Verifier detects cryptographic hash mismatch; rejects at S15-00 | S15-15 |
| **Mutation 5** | Corrupted coefficient ($\alpha = 2.0$) in `protocol.json` | Verifier detects configuration mismatch against frozen $\Theta_0$; rejects at S15-01 | S15-15 |
| **Mutation 6** | Missing `production_route_boundary_passed` evidence in `test_results.json` | Verifier detects missing required boundary verification field; rejects at S15-10 | S15-15 |
| **Mutation 7** | Mutation count mismatch in `test_results.json` | Verifier detects mutation count discrepancy; rejects at S15-15 | S15-15 |
| **Mutation 8** | Malformed non-numeric string in numeric field in `test_results.json` | Verifier safely fails gate without unhandled crash; rejects at S15-06 | S15-15 |
| **Mutation 9** | Summary status altered to `FAILED` in `summary.json` | Verifier detects cross-artifact evaluation status inconsistency; rejects at S15-00 | S15-15 |
| **Mutation 10** | Summary phase altered to `C11.14` in `summary.json` | Verifier detects cross-artifact phase mismatch; rejects at S15-00 | S15-15 |
| **Mutation 11** | Summary `gates_passed` altered to `14/15` in `summary.json` | Verifier detects cross-artifact gate count discrepancy; rejects at S15-00 | S15-15 |
| **Mutation 12** | Null `artifact_hashes` dictionary in `freeze_manifest.json` | Verifier detects schema structural invalidity; rejects at S15-00 | S15-15 |
| **Mutation 13** | Malformed root JSON list instead of dictionary in `protocol.json` | Verifier detects root schema failure; rejects at S15-00 | S15-15 |
| **Mutation 14** | Scorecard criterion flag inverted in `summary.json` contradicting `test_results.json` | Verifier detects direct cross-artifact scorecard contradiction; rejects at S15-00 | S15-15 |
