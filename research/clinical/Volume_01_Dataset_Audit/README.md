# Research Volume 01: Clinical Dataset & Task Audit

## Overview
This volume documents the formal clinical data audit, provenance verification, schema inspection, target formulation, feature taxonomy categorization, missingness profiling, identifier analysis, leakage risk register, clinical limitations, cryptographic fingerprint freeze, and automated verification gate for the UCI Diabetes 130-US Hospitals dataset (`datasets/clinical/`).

## Contents

- [01_introduction.md](file:///d:/FusionMedAI/research/clinical/Volume_01_Dataset_Audit/01_introduction.md): Scope, objectives, audit principles, and C1 boundaries.
- [02_dataset_source.md](file:///d:/FusionMedAI/research/clinical/Volume_01_Dataset_Audit/02_dataset_source.md): Primary UCI source, Cerner Health Facts provenance, extraction criteria, and file inventory.
- [03_schema_audit.md](file:///d:/FusionMedAI/research/clinical/Volume_01_Dataset_Audit/03_schema_audit.md): Raw 50 columns, dtypes, cardinality, unique value counts, and zero-variance columns.
- [04_target_definition.md](file:///d:/FusionMedAI/research/clinical/Volume_01_Dataset_Audit/04_target_definition.md): Exact `readmitted` target, 30-day early readmission binary formulation, and class prevalence.
- [05_feature_taxonomy.md](file:///d:/FusionMedAI/research/clinical/Volume_01_Dataset_Audit/05_feature_taxonomy.md): Exhaustive partitioning of 50 features into 9 clinical and operational domains.
- [06_missingness_profile.md](file:///d:/FusionMedAI/research/clinical/Volume_01_Dataset_Audit/06_missingness_profile.md): Separation of `'?'` missing data, informative lab test `'None'`, and administrative missingness codes.
- [07_identifier_analysis.md](file:///d:/FusionMedAI/research/clinical/Volume_01_Dataset_Audit/07_identifier_analysis.md): `encounter_id` and `patient_nbr` roles, single vs repeat encounter dynamics, and patient grouping.
- [08_leakage_risk_register.md](file:///d:/FusionMedAI/research/clinical/Volume_01_Dataset_Audit/08_leakage_risk_register.md): 5 leakage risk categories, expired/hospice status audit, and mitigation protocols.
- [09_clinical_limitations.md](file:///d:/FusionMedAI/research/clinical/Volume_01_Dataset_Audit/09_clinical_limitations.md): Explicit catalog of retrospective EHR limitations for peer-reviewed publication.
- [10_dataset_freeze.md](file:///d:/FusionMedAI/research/clinical/Volume_01_Dataset_Audit/10_dataset_freeze.md): Frozen dataset specifications, SHA-256 hashes, environment, and data contract.
- [verify_dataset_audit.py](file:///d:/FusionMedAI/verification/clinical/data/verify_dataset_audit.py): Automated 11-gate programmatic verification script.

## Verification Gate Summary

```
======================================================================
Dataset identity              PASS
Dataset integrity             PASS
Schema verified               PASS
Target frozen                 PASS
Feature taxonomy              PASS
Missingness documented        PASS
Identifiers classified        PASS
Leakage candidates identified PASS
Clinical limitations          PASS
Dataset fingerprint frozen    PASS
Reproducibility               PASS
──────────────────────────────────────────────────────────────────────
C1 OVERALL STATUS             PASS
======================================================================
```
