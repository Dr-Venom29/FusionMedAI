# Research Volume 02: Clinical Data Pipeline Construction

## Overview
This volume documents the structural analysis of patient and encounter hierarchies, repeat-encounter dynamics, temporal encounter sequencing, discharge-boundary predictor availability, cohort eligibility construction, splitting strategy evaluation, canonical patient-grouped train/val/test splitting, zero-leakage verification, and the foundational pipeline contract for the Clinical Modality of **FusionMedAI**.

## Contents

- [01_patient_encounter_structure.md](01_patient_encounter_structure.md): Patient vs encounter hierarchy ($101,766$ encounters across $71,518$ patients).
- [02_repeat_encounter_analysis.md](02_repeat_encounter_analysis.md): Concentration of repeat visits ($23.45\%$ patients generate $46.20\%$ encounters) and within-patient target transitions.
- [03_temporal_ordering_audit.md](03_temporal_ordering_audit.md): Monotonic `encounter_id` sequence analysis and chronological drift isolation.
- [04_predictor_availability_audit.md](04_predictor_availability_audit.md): Formal audit of candidate predictors against the discharge planning decision boundary.
- [05_cohort_eligibility.md](05_cohort_eligibility.md): Isolation of $2,423$ expired/hospice cases to `interim/cohort_eligibility.csv` and establishment of the $99,343$-encounter eligible cohort.
- [06_split_strategy_study.md](06_split_strategy_study.md): Comparative evaluation of splitting strategies and justification for patient-grouped partitioning.
- [07_patient_grouped_split.md](07_patient_grouped_split.md): Canonical $70\% / 15\% / 15\%$ patient-grouped split (Seed $42$) and artifact generation.
- [08_leakage_verification.md](08_leakage_verification.md): Automated verification of zero patient overlap, zero identifier leakage, and zero cohort ineligibility in splits.
- [09_pipeline_contract.md](09_pipeline_contract.md): Strict preprocessing boundaries and training-only estimator fitting rules for Phase C3+.

---

## Phase C2 Verification Gate

```
==================================================
FusionMedAI: Phase C2 Verification Gate
==================================================

[1] Patient / Encounter Structure        PASS
[2] Repeat-Encounter Analysis            PASS
[3] Temporal Structure Audit             PASS
[4] Predictor Availability Audit         PASS
[5] Cohort Eligibility                   PASS
[6] Split Strategy                       PASS
[7] Patient-Grouped Split                PASS
[8] Patient Overlap Check                PASS
[9] Leakage Verification                 PASS
[10] Deterministic Reproducibility       PASS
[11] Pipeline Contract                   PASS
--------------------------------------------------
C2 STATUS                                PASS
==================================================
```
