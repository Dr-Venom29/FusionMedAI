# Chapter 09 — Phase 10.8 Acceptance Criteria & Verification Protocol

## 1. Phase 10.8 Acceptance Checklist

| Item | Requirement Description | Verification Method | Status |
| :---: | :--- | :--- | :---: |
| 1 | Frozen B3 Checkpoint Verified | Inspect `experiments/foot/.../best_model.pt` | Verified |
| 2 | Frozen Vector Scaling Artifact Verified | Inspect `experiments/foot/final_model/calibration.json` | Verified |
| 3 | MC Dropout Implementation Verified | `enable_foot_mc_dropout()` sets Dropout to `.train()`, BatchNorm to `.eval()` | Verified |
| 4 | Pass-Count Convergence Evaluated | Validation set ($N=1,006$) evaluated for $N \in \{5 \dots 30\}$ | Verified |
| 5 | Final $N^{*}$ Selected from Convergence | $N^{*}=10$ selected via rule-based stabilization ($\Delta H \le 10^{-3}$) | Verified |
| 6 | Full 1,006-Image Test Evaluation Completed | Held-out test split evaluated under Option B pipeline | Verified |
| 7 | Predictive Variance Calculated | Class variance averaged per sample | Verified |
| 8 | Predictive Entropy Calculated | Shannon entropy of mean probabilities | Verified |
| 9 | Mutual Information Calculated | Total entropy minus expected entropy | Verified |
| 10 | Error-Detection Metrics Calculated | AUROC & AUPRC computed for error detection | Verified |
| 11 | Risk-Coverage Analysis Completed | Evaluated at 100%, 90%, 80%, 70%, 60%, 50% coverage | Verified |
| 12 | G2/G3 Boundary Analysis Completed | Sub-cohort statistics computed for G2 vs G3 | Verified |
| 13 | Class-Wise Uncertainty Analysis Completed | Disaggregated for G1, G2, G3, G4 | Verified |
| 14 | High-Uncertainty Error Analysis Completed | 45 unique cases extracted and Grad-CAM overlaid | Verified |

| 15 | Results Reproducible | Scripts run deterministically with seed 42 | Verified |
| 16 | Uncertainty Configuration Frozen | Written to `experiments/foot/final_model/uncertainty.json` | Verified |
| 17 | Verification Script Passes | 12-point automated verification suite passes (12/12 PASS) | Verified |
| 18 | Research Documentation Updated | Volume VIII Chapters 01–09 documented | Verified |

---

## 2. 12-Point Automated Verification Suite

The verification suite (`verification/foot/model/verify_uncertainty.py`) executes 12 explicit checks:

1. `Test 1`: Canonical B3 Checkpoint Loading
2. `Test 2`: Frozen Vector Scaling Artifact Loading
3. `Test 3`: MC Dropout Stochasticity Verification
4. `Test 4`: Deterministic Evaluation Mode Verification
5. `Test 5`: Probabilities Finite and Normalization ($ \sum p_k = 1.0 $)
6. `Test 6`: Entropy Bounds Check ($ 0 \le H(p) \le \log(4) $)
7. `Test 7`: Predictive Variance Non-Negativity ($ \text{Var}(p) \ge 0 $)
8. `Test 8`: Epistemic Mutual Information Bounds ($ MI \ge 0 $)
9. `Test 9`: Stochastic Pass Count Specification ($ N=10 $)
10. `Test 10`: Label Independence Protocol Check (unsupervised uncertainty computation)
11. `Test 11`: Risk-Coverage Rejection Monotonicity Check
12. `Test 12`: Uncertainty Array Serialization & Reload Verification

---

## 3. Phase 10.8 Formal Sign-Off

All 18 acceptance criteria and all 12 automated verification checks (`12/12 PASS`) have been satisfied. Phase 10.8 — Prediction Uncertainty Estimation for Diabetic Foot Ulcer Wagner Classification is officially **COMPLETE**.
