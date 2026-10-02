# Research Volume 02: Foot Data Pipeline & Group-Based Partitioning

## Overview
This volume documents the design, implementation, and verification of the leakage-controlled data pipeline for the Diabetic Foot Ulcer (DFU) Wagner 4-Class modality (`src/foot/data/`).

## Contents

- [01_Objectives.md](./01_Objectives.md): Pipeline goals, architectural constraints, and requirements.
- [02_Exact_Duplicate_Resolution.md](./02_Exact_Duplicate_Resolution.md): SHA-256 duplicate resolution (10,050 canonical images, 12 excluded).
- [03_Canonical_Manifest.md](./03_Canonical_Manifest.md): Authoritative modeling manifest `canonical_manifest.csv`.
- [04_Source_Group_Construction.md](./04_Source_Group_Construction.md): Construction of 1,770 source-image groups.
- [05_Near_Duplicate_Analysis.md](./05_Near_Duplicate_Analysis.md): Categorization of near-duplicate relationships (2,452 TYPE 1 pairs isolated).
- [06_Group_Stratified_Splitting.md](./06_Group_Stratified_Splitting.md): Deterministic 80/10/10 group-stratified split on source_image_id.
- [07_Split_Verification.md](./07_Split_Verification.md): Leakage, integrity, and class distribution verification.
- [08_Dataset_Implementation.md](./08_Dataset_Implementation.md): `FootDFUDataset` PyTorch dataset class in `src/foot/data/dataset.py`.
- [09_Image_Transforms.md](./09_Image_Transforms.md): Candidate transformation pipelines in `src/foot/data/transforms.py`.
- [10_DataLoader.md](./10_DataLoader.md): DataLoader creation (`create_foot_dataloaders`) in `src/foot/data/dataloader.py`.
- [11_Pipeline_Verification.md](./11_Pipeline_Verification.md): End-to-end pipeline verification and backprop test in `src/foot/data/verify_pipeline.py`.
- [12_Acceptance.md](./12_Acceptance.md): Phase 10.2 Acceptance Gate sign-off.
