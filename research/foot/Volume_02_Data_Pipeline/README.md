# Research Volume 02: Foot Data Pipeline & Group-Based Partitioning

## Overview
This volume documents the design, implementation, and verification of the leakage-controlled data pipeline for the Diabetic Foot Ulcer (DFU) Wagner 4-Class modality (`src/foot/data/`).

## Contents

- [01_Objectives.md](file:///d:/FusionMedAI/research/foot/Volume_02_Data_Pipeline/01_Objectives.md): Pipeline goals, architectural constraints, and requirements.
- [02_Exact_Duplicate_Resolution.md](file:///d:/FusionMedAI/research/foot/Volume_02_Data_Pipeline/02_Exact_Duplicate_Resolution.md): SHA-256 duplicate resolution (10,050 canonical images, 12 excluded).
- [03_Canonical_Manifest.md](file:///d:/FusionMedAI/research/foot/Volume_02_Data_Pipeline/03_Canonical_Manifest.md): Authoritative modeling manifest `canonical_manifest.csv`.
- [04_Source_Group_Construction.md](file:///d:/FusionMedAI/research/foot/Volume_02_Data_Pipeline/04_Source_Group_Construction.md): Construction of 1,770 source-image groups.
- [05_Near_Duplicate_Analysis.md](file:///d:/FusionMedAI/research/foot/Volume_02_Data_Pipeline/05_Near_Duplicate_Analysis.md): Categorization of near-duplicate relationships (2,452 TYPE 1 pairs isolated).
- [06_Group_Stratified_Splitting.md](file:///d:/FusionMedAI/research/foot/Volume_02_Data_Pipeline/06_Group_Stratified_Splitting.md): Deterministic 80/10/10 group-stratified split on source_image_id.
- [07_Split_Verification.md](file:///d:/FusionMedAI/research/foot/Volume_02_Data_Pipeline/07_Split_Verification.md): Leakage, integrity, and class distribution verification.
- [08_Dataset_Implementation.md](file:///d:/FusionMedAI/research/foot/Volume_02_Data_Pipeline/08_Dataset_Implementation.md): `FootDFUDataset` PyTorch dataset class in `src/foot/data/dataset.py`.
- [09_Image_Transforms.md](file:///d:/FusionMedAI/research/foot/Volume_02_Data_Pipeline/09_Image_Transforms.md): Candidate transformation pipelines in `src/foot/data/transforms.py`.
- [10_DataLoader.md](file:///d:/FusionMedAI/research/foot/Volume_02_Data_Pipeline/10_DataLoader.md): DataLoader creation (`create_foot_dataloaders`) in `src/foot/data/dataloader.py`.
- [11_Pipeline_Verification.md](file:///d:/FusionMedAI/research/foot/Volume_02_Data_Pipeline/11_Pipeline_Verification.md): End-to-end pipeline verification and backprop test in `src/foot/data/verify_pipeline.py`.
- [12_Acceptance.md](file:///d:/FusionMedAI/research/foot/Volume_02_Data_Pipeline/12_Acceptance.md): Phase 10.2 Acceptance Gate sign-off.
