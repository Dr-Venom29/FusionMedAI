# Research Volume 03: Exploratory Data Analysis & Feature Representation

## Overview
This volume documents the formal exploratory data analysis (EDA), split and target distribution auditing, missingness mechanism profiling, numerical and categorical feature evaluation, clinical feature-group characterization, ICD-9 diagnosis chapter mapping, pharmacotherapy exposure modeling, pairwise association and redundancy analysis, clinical outlier auditing, subgroup risk profiling, and the frozen feature representation contract for the Clinical Modality of **FusionMedAI**.

## Contents

- [01_split_target_distribution.md](01_split_target_distribution.md): Audit of canonical $70/15/15$ train/val/test splits and target stability ($11.39\% \pm 0.28\%$).
- [02_missingness_structure.md](02_missingness_structure.md): Operational missingness profiles (`'?'` unrecorded data vs `'None'` informative lab test absence).
- [03_numerical_feature_analysis.md](03_numerical_feature_analysis.md): Distributional analysis, skewness, and class separation across 8 numerical features.
- [04_categorical_feature_analysis.md](04_categorical_feature_analysis.md): Cardinality audit, frequency distribution, and high-cardinality management for `medical_specialty`.
- [05_clinical_feature_groups.md](05_clinical_feature_groups.md): Evaluation of 9 clinical and operational feature domains.
- [06_diagnosis_representation.md](06_diagnosis_representation.md): 9-Chapter ICD-9 clinical disease mapping across `diag_1`, `diag_2`, and `diag_3`.
- [07_medication_treatment_representation.md](07_medication_treatment_representation.md): 4-level exposure modeling ($0, 1, 2, 3$) for 21 active anti-diabetic medications.
- [08_feature_association_redundancy.md](08_feature_association_redundancy.md): Spearman rank correlation, categorical contingency, and collinearity registers.
- [09_outlier_distribution_analysis.md](09_outlier_distribution_analysis.md): Clinical plausibility audit of extreme values (zero sample deletions justified).
- [10_subgroup_analysis.md](10_subgroup_analysis.md): Exploratory readmission risk gradients across age, race, gender, and clinical subgroups.
- [11_feature_representation_contract.md](11_feature_representation_contract.md): Frozen attribute-by-attribute preprocessing and encoding contract.

---

## Phase C3 Verification Gate

```
==================================================
FusionMedAI: Phase C3 Verification Gate
==================================================

[ 1] Split row counts                     PASS
[ 2] Target prevalence                    PASS
[ 3] Complete patient isolation           PASS
[ 4] Missingness artifact                 PASS
[ 5] Numerical statistics artifact        PASS
[ 6] Categorical statistics artifact      PASS
[ 7] Clinical feature-group artifact      PASS
[ 8] Diagnosis mapping artifact           PASS
[ 9] Medication statistics artifact       PASS
[10] Association artifact                 PASS
[11] Outlier artifact                     PASS
[12] Subgroup artifact                    PASS
[13] Feature representation contract      PASS
[14] Train-only provenance                PASS
[15] Artifact integrity / hashes          PASS
[16] Frozen artifact reproducibility      PASS
--------------------------------------------------
C3 STATUS                                PASS
==================================================
```
