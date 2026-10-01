# Document 01: Robustness Auditing Protocol & Evaluation Invariants

## 1. Upstream Frozen Model Contract (Phases C5–C8)

Phase C9 is strictly designed as an **internal out-of-distribution stress testing and reliability auditing phase**. In adherence to empirical governance protocols, no machine learning parameters are learned, modified, or tuned during Phase C9.

### Frozen System Specification:
1. **Model Architecture**: CatBoostClassifier (Symmetric Oblivious Decision Trees, Depth 4, Learning Rate $0.1383$, $350$ iterations, $L_2$ Leaf Reg $2.911$, Subsample $0.655$, Random Seed $42$).
2. **Feature Representation**: $D = 119$ clinical features via `ClinicalPreprocessor(scale_numerical=True)`.
3. **Calibration Protocol**: Post-hoc Isotonic Regression calibrator strictly learned on the validation partition ($N=14,911$).
4. **Uncertainty Architecture**: 50-member Bootstrap CatBoost Ensemble ($M=50$, random seeds $43 \dots 92$).
5. **Operating Threshold**: Institutional clinical baseline $\theta = 0.20$.
6. **Data Partitions**: Canonical three-way split ($N_{\text{train}}=69,519$, $N_{\text{val}}=14,911$, $N_{\text{test}}=14,913$).

---

## 2. Shift Invariant Rules

To prevent empirical contamination and maintain statistical validity:

1. **Zero Retraining Rule**:
   The CatBoost backbone and bootstrap members are never retrained on shifted or perturbed data. The model evaluated under distribution shift is identical to the model evaluated in nominal test conditions.
2. **Zero Calibration Re-fitting**:
   The Isotonic calibration mapping remains fixed as fitted on nominal validation data. We test whether the *nominal calibration mapping degrades* under shift, rather than adapting calibrators to test shifts.
3. **Zero Test Partition Re-sampling**:
   All synthetic perturbations (missingness, targeted masking, composition weighting) are applied strictly to the out-of-sample locked test partition ($N=14,913$) or subsets thereof.

---

## 3. Pre-Registered Acceptance & Robustness Criteria

Before analyzing test results, the following evaluation criteria were pre-registered:

| Evaluation Dimension | Pre-Registered Metric Threshold | Clinical Rationale |
| :--- | :--- | :--- |
| **Moderate Missingness Tolerance** | $\Delta \text{ROC-AUC} \ge -0.050$ at $+25\%$ MCAR | Model should retain discriminative capability under moderate missing data. |
| **Severe Missingness Degradation** | Monotonic $\Delta \text{ROC-AUC} \le 0$ with $\sigma_p$ growth | Total performance loss under $+50\%$ MCAR must be transparently reported. |
| **Uncertainty Response Criterion** | Mean $\sigma_p$ must increase by $\ge +30\%$ at $+25\%$ MCAR | Epistemic uncertainty must actively signal information loss. |
| **Demographic Calibration Audit** | Calibration slope $0.65 \le \beta \le 1.35$ across major strata | Risk probabilities must maintain proper directionality across cohorts. |
| **Temporal Stability** | $\Delta \text{ROC-AUC} \ge -0.030$ between early and late eras | Clinical predictor must not experience rapid longitudinal collapse. |
| **Selective Classification Validity** | $\text{Error}_{80\%} < \text{Error}_{100\%}$ across all shift scenarios | Uncertainty-based triage must remain effective even under distribution shift. |
| **Failure Transparency** | Complete accounting of high-confidence silent errors | All unalerted false predictions must be cataloged and documented. |

---

## 4. Scientific Scope & Governance Non-Claims

1. **No Absolute Fairness Claim**:
   Evaluating performance across demographic cohorts (e.g., Male vs. Female, African American vs. Caucasian) audits empirical stability and calibration parity under retrospective conditions. Subgroup differences (e.g., younger vs. older discrimination gradients) are documented characteristics of the dataset, not mathematical proofs of algorithmic equity or bias elimination.
2. **No External Dataset Generalization**:
   Phase C9 stress tests the locked test partition and its perturbed/stratified variants. It does not establish robustness on external health systems or independent hospital cohorts. Cross-dataset generalization is strictly evaluated in Phase C10.
3. **Synthetic MCAR Boundary**:
   MCAR perturbations simulate controlled information degradation to evaluate algorithmic sensitivity; they do not represent real-world clinical missing-data mechanisms.
4. **Sampling Variance in Subgroups**:
   Subgroup estimates (e.g., Frequent Inpatient $\ge 3$ with $N=986$, African American with $N=2,775$) inherently possess higher sampling variance than the full $N=14,913$ test cohort.
5. **No Autonomous Deployment Mandate**:
   Robustness under simulated distribution shift does not imply readiness for unmonitored bedside autonomy. All outputs remain bounded as Clinical Decision Support (CDS) alerts requiring clinician oversight.
