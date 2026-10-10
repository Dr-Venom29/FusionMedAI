# Volume 15: ACARA-U Residual Router Sanity & Monotonicity Analysis

## Overview

Volume 15 provides a comprehensive mathematical sanity, invariant verification, single-input monotonicity, and pipeline contract audit of the **ACARA-U v2 Dynamic Multimodal Router** under the frozen reference parameter configuration $\Theta_0 = (1.0, 1.5, 1.0, 0.5)$ and frozen uncertainty discount $\delta^* = 0.10$.

This analysis confirms that the ACARA-U routing engine satisfies all theoretical mathematical properties, handles degenerate and corrupt inputs gracefully, executes production downstream fused-risk and DCRI contracts cleanly, and maintains strict decoupling between router weight allocation and downstream fused risk responses.

> **Volume 15 Status:** 🟢 SEALED & VERIFIED (16/16 Gates Passed)

---

## Key Research Findings

1. **Strict Monotonic Response Across Multi-Modality Regimes:**
   In all multi-modality active sets ($|\mathcal{A}| \ge 2$), increasing confidence ($C_i$), reliability ($R_i$), or quality ($Q_i$) strictly increases channel decision weight $w_i$, while increasing predictive uncertainty ($U_i$) strictly decreases $w_i$ (100% pass across 576 trials).

2. **Single-Modality Unity Invariance ($w_i \equiv 1.0$):**
   In single-modality regimes (`R`, `F`, `C`), $w_i \equiv 1.000000$ remains invariant under all input perturbations, as required by softmax normalization over a singleton active set. Downstream DCRI discounting provides conservative compensation.

3. **Machine-Precision Simplex & Invariant Conservation:**
   The router preserves the probability simplex ($\sum w_i = 1.000000$) with a maximum deviation of $1.11 \times 10^{-16}$, enforces hard availability masking ($A_j=0 \implies w_j=0.000000$), satisfies softmax shift invariance ($\Delta w \le 7.11 \times 10^{-15}$ across logit offsets $[-500, 500]$ and $\Delta w \le 1.11 \times 10^{-16}$ in uniform confidence route shift), and achieves order independence in the reference normalization kernel across all 6 channel permutations.

4. **Fail-Closed Degenerate Handling & Robust Validation:**
   Encountering an empty modality set ($A = \emptyset$) safely triggers `NO_MODALITY_AVAILABLE` with zero weight allocation and zero active modalities. All 19 invalid-input test cases (NaN, Inf, out-of-range scalars, reliability mismatches, invalid coefficients) are strictly rejected with precise concrete exceptions (`RouterContractValidationError` and `RouterCoefficientError`, subclasses of `ValueError`).

5. **Decoupling of Router Weights and Downstream Fused Risk:**
   An increase in modality weight $w_i$ does not automatically increase fused risk $R_{\text{fusion}} = \sum w_j r_j$. The direction of risk change is governed by whether $r_i$ is greater or less than the weighted average of other channels. Downstream evaluations directly exercise production DCRI modules.

---

## Chapter Directory

| Chapter | Title | Primary Focus |
| :--- | :--- | :--- |
| **[Chapter 01](01_Protocol.md)** | Protocol & Acceptance Criteria | Research hypotheses, mathematical formulation, and 16 acceptance criteria. |
| **[Chapter 02](02_Test_Matrix.md)** | Test Matrix & Methodology | Single-input perturbation grids, invariant test suites, and fault injection matrices. |
| **[Chapter 03](03_Results.md)** | Experimental Results & Findings | Scorecard summary, 576 monotonicity trial results, invariant metrics, and pipeline decoupling data. |
| **[Chapter 04](04_Failure_Analysis.md)** | Boundary & Decoupling Analysis | Singleton invariance derivation, mathematical proof of risk decoupling, and mutation audit. |
| **[Chapter 05](05_Freeze_Report.md)** | Freeze Declaration & Certification | Formal certification seal, frozen execution envelope, and SHA-256 artifact manifest. |

---

## Verification Summary

- **Deep Verification Gates:** 16 / 16 Passed (100.0%)
- **Unit & Invariant Test Suite:** 36 / 36 Tests Passed (including 14 verifier mutation tests)
- **Regression Suite:** 22 / 22 C11.14 DCRI Policy Analysis Tests Passed (0 failures, 0 errors, 0 skips)
- **Mutation Fault Detection:** 14 / 14 Injected Faults Detected & Rejected