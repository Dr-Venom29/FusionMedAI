# Research Document 10: Clinical Explainability Synthesis & Governance Sign-Off

## 1. Synthesis of Explainability Findings

Phase C6 conducted a comprehensive post-hoc TreeSHAP interpretability study on the frozen **CatBoost HPO candidate** ($0.6504$ Test ROC-AUC, $0.2063$ Test PR-AUC) across $14,913$ test encounters ($D=119$).

### Key Algorithmic & Attribution Findings:
1. **Prior Utilization & Acute Complexity Dominance**:
   - `number_inpatient` is the single most dominant predictor ($22.43\%$ of total mean absolute SHAP attribution), with higher values associated with higher model-attributed risk ($r = +0.9531$).
   - Acute complexity (`time_in_hospital`, `number_diagnoses`, `num_medications`) contributes an additional $15.21\%$ share.
2. **Clinical Taxonomy Structure**:
   - **Prior Healthcare Utilization** ($26.22\%$) and **Acute Clinical Complexity** ($21.23\%$) together account for **$47.45\%$** of total mean absolute SHAP attribution.
   - **ICD-9 Diagnosis Chapters** ($15.25\%$) and **Encounter Context** ($13.51\%$) provide additional aggregate predictive attribution.
   - **Demographics** have minimal direct attribution ($1.91\%$), with similar attribution magnitudes observed across female and male cohorts.
3. **Data Contract & Leakage Review**:
   - Variables in the $D=119$ space were verified against the Phase C1–C4 data contracts to be documented prior to hospital discharge sign-off.
   - No target-encoding leakage or post-discharge future information exists in the feature representation.
4. **Attribution Ranking Stability**:
   - Validation vs. Test ranking correlation achieved $\rho = 0.9994$ ($p = 3.86 \times 10^{-172}$) with $100\%$ Top-20 overlap.

---

## 2. Integrity & Sanity Audit Checklist

| Audit Question | Verified Result | Assessment |
| :--- | :---: | :---: |
| Is any feature derived from the 30-day readmission outcome? | No | PASS |
| Are all features available prior to discharge order sign-off? | Yes | PASS |
| Do feature directionalities show consistent descriptive empirical correlations? | Yes | PASS |
| Were demographic attributions minimal and balanced across reported cohorts? | Yes (1.91% share, $\Delta < 0.002$) | PASS |
| Is global feature attribution ranking stable across validation and test splits? | Yes ($\rho=0.9994$) | PASS |

---

## 3. C6 Limitations & Non-Claims

1. **Non-Causal Attribution**: SHAP values describe model attribution in log-odds space and do not establish biological causes, pathophysiological mechanisms, or clinical causality.
2. **Error-Cohort Attribution Scope**: Error-cohort SHAP attributions describe how features influenced the model output; they do not establish why an observed outcome occurred or did not occur in clinical reality.
3. **Subgroup Attribution vs. Algorithmic Fairness**: Similar subgroup SHAP distributions reflect attribution consistency across reported cohorts, but do not constitute a complete algorithmic fairness audit (e.g., error-rate parity, calibration parity, or absence of proxy effects).
4. **Stability Scope**: Validation–test attribution stability demonstrates ranking consistency across partitions, but does not independently establish absence of overfitting or clinical effectiveness.
5. **Directionality Interpretation**: Directionality correlations are descriptive summaries of linear association between feature values and SHAP contributions; they should not be interpreted as causal dose-response relationships.
6. **Clinical Hypotheses**: Clinical interpretations discussed in this volume are exploratory hypotheses for downstream investigation rather than treatment recommendations or clinical practice guidelines.

---

## 4. Phase C6 Governance Sign-Off

```
===========================================================================
FusionMedAI: Phase C6 Clinical Model Explainability Governance Sign-Off
===========================================================================
[✓] Frozen CatBoost HPO Model (Trial 2, Depth 4) Formally Evaluated
[✓] TreeSHAP Computed on Validation (14,911) & Test (14,913) Matrices
[✓] Global Feature Rankings (Top 20 / Top 30) Exported and Verified
[✓] Feature Directionalities & Dependence Profiles Characterized
[✓] 8 Clinical Taxonomy Groups Aggregated and Analyzed
[✓] Local Encounter Explanations Generated for 5 Risk Cohorts
[✓] Error Attributions (FP vs. FN at θ=0.20) Fully Documented
[✓] Subgroup Attribution Consistency Verified Across Reported Cohorts
[✓] Validation vs. Test Stability Certified (ρ = 0.9994, 100% Top-20)
[✓] Cryptographic Manifest Hashed and Locked (c6_manifest.json)
---------------------------------------------------------------------------
PHASE C6 EXPLAINABILITY STATUS:                                 COMPLETE
CLINICAL VALIDATION STATUS:                                     PENDING (Future Work)
NEXT CLINICAL PHASE:                                            C7 (Probability Calibration)
===========================================================================
```
