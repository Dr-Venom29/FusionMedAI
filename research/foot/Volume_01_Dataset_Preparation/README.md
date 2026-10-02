# Research Volume 01: Foot Dataset Preparation & Audit

## Overview
This volume documents the acquisition, inventory, label verification, image decodability integrity, property characterization, duplicate analysis, patient-case leakage investigation, provenance audit, and formal quality decision for the Diabetic Foot Ulcer (DFU) Wagner 4-Class dataset (`datasets/foot/raw/`).

## Contents

- [01_Objectives.md](./01_Objectives.md): Audit goals, fail-fast requirements, and scope.
- [02_Dataset_Selection.md](./02_Dataset_Selection.md): Dataset source identification and raw layout.
- [03_Dataset_Inventory.md](./03_Dataset_Inventory.md): Quantitative file count, size, and format inventory.
- [04_Label_Definition.md](./04_Label_Definition.md): Wagner-Meggitt classification system mapping and clinical definitions.
- [05_Data_Integrity.md](./05_Data_Integrity.md): Decodability, corruption checks, and dimension validation.
- [06_Image_Property_Audit.md](./06_Image_Property_Audit.md): Resolution and channel statistics (Observed Mean & Std).
- [07_Duplicate_and_Leakage_Audit.md](./07_Duplicate_and_Leakage_Audit.md): Cryptographic SHA-256 duplicate detection and dHash near-duplicate analysis.
- [08_Source_Group_Analysis.md](./08_Source_Group_Analysis.md): Extraction of 1,770 source-image groups and offline expansion ratio.
- [09_Quality_Decision.md](./09_Quality_Decision.md): Synthesis, rationale, limitations, and CONDITIONAL PASS decision.
- [10_Acceptance.md](./10_Acceptance.md): Phase 10.1 acceptance sign-off.
