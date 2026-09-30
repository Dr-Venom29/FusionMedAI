# Research Volume 01: Clinical Dataset & Task Audit

## Overview
This volume documents the formal clinical data audit, provenance verification, schema inspection, target formulation, feature taxonomy categorization, missingness profiling, identifier analysis, leakage risk register, clinical limitations, cryptographic fingerprint freeze, and automated verification gate for the UCI Diabetes 130-US Hospitals dataset (`datasets/clinical/`).

## Contents

- [01_introduction.md](01_introduction.md): Scope, objectives, audit principles, and C1 boundaries.
- [02_dataset_source.md](02_dataset_source.md): Primary UCI source, Cerner Health Facts provenance, extraction criteria, and file inventory.
- [03_schema_audit.md](03_schema_audit.md): Raw 50 columns (47 predictive attributes + 2 identifiers + 1 target), dtypes, cardinality, unique value counts, and zero-variance columns.
- [04_target_definition.md](04_target_definition.md): Exact `readmitted` target, 30-day early readmission binary formulation, class prevalence, and discharge-time prediction boundary.
- [05_feature_taxonomy.md](05_feature_taxonomy.md): Exhaustive partitioning of 50 columns into 9 clinical and operational domains.
- [06_missingness_profile.md](06_missingness_profile.md): Separation of `'?'` missing data, informative lab test `'None'`, and administrative missingness codes.
- [07_identifier_analysis.md](07_identifier_analysis.md): `encounter_id` and `patient_nbr` roles, single vs repeat encounter dynamics, and patient grouping.
- [08_leakage_risk_register.md](08_leakage_risk_register.md): 5 risk categories, expired/hospice structural determinism audit ($N=2,423$), and mitigation protocols.
- [09_clinical_limitations.md](09_clinical_limitations.md): Explicit catalog of retrospective EHR limitations for peer-reviewed publication.
- [10_dataset_freeze.md](10_dataset_freeze.md): Frozen dataset specifications, SHA-256 hashes, environment, and data contract.
- [verify_dataset_audit.py](../../../verification/clinical/data/verify_dataset_audit.py): Automated 11-gate programmatic verification script.

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
----------------------------------------------------------------------
C1 OVERALL STATUS             PASS
======================================================================
```
