# Chapter 2: Research Objectives and Contributions

## 2.1 Objectives
The primary objective of this phase is to establish a **fully reproducible, automated, and reliable dataset preparation and verification pipeline** for the retinal fundus images within the FusionMedAI framework. This pipeline acts as a robust and standardized data ingestion gateway, designed to ensure that every sample fed into subsequent deep learning models is structurally sound, correctly labeled, and formatted.

**Secondary Objectives:**
- **Ensure Data Integrity**: Provide programmatic verification that all local raw files are complete, uncorrupted, and perfectly synchronized with their corresponding tabular index records.
- **Eliminate Corrupted Samples**: Detect and flag unopenable or truncated image files that would otherwise cause batch load crashes or training instability.
- **Produce Reusable Metadata**: Produce metadata that supports downstream exploratory data analysis, preprocessing optimization, quality assessment, and reproducible model development.
- **Improve Experiment Reproducibility**: Enforce complete reproducibility across multiple development environments by centralizing all hyperparameters, directory paths, and random seeds within a single configuration script.
- **Standardize Project Organization**: Establish a structured directory layout that separates immutable raw clinical data from transient preprocessed data, metadata reports, and experimental outputs.

## 2.2 Research Questions
To validate the engineering and design choices in this dataset preparation phase, the project addresses the following core research questions:

1. **How can dataset integrity be automatically verified in a medical image pipeline to prevent silent training failures?**
   - *Investigation*: Designing an automated suite of structural and logical checks (CSV existence, folder layout, file corruption, label boundaries) that runs in a single script and returns unified error logs.
   
2. **Which metadata attributes most effectively support subsequent preprocessing, EDA, and model development?**
   - *Investigation*: Determining which per-image metrics (width, height, aspect ratio, channels, file size, ground-truth diagnosis) are crucial for optimizing data augmentation strategies, batch padding, and preprocessing pipelines without needing to read raw files repeatedly.

3. **How can reproducibility be improved through standardized project organization, configuration management, and automated verification?**
   - *Investigation*: Evaluating the impact of a centralized, read-only configuration module combined with a strict separation of raw datasets, interim files, and generated metadata on preventing code drift.

## 2.3 Expected Outcomes
The execution of this phase is designed to yield the following expected deliverables:
- **Verified dataset**: A dataset with programmatically audited image paths.
- **Zero corrupted files**: Complete isolation of problematic files to prevent runtime training exceptions.
- **Standardized metadata**: Comprehensive, pre-computed data parameters detailing image metrics and class labels.
- **Reproducible directory structure**: A strict separation of read-only raw files, metadata reports, and transient outputs.
- **Centralized configuration**: Unified paths, random seeds, and class labels declared in a single source of truth.
- **Ready-to-use dataset pipeline**: A validated data entry point prepared for downstream preprocessing.

## 2.4 Research Contributions
This work introduces several key methodological and engineering contributions to the diabetic retinopathy classification pipeline within the FusionMedAI framework.

1. **Automated Medical Dataset Verification Framework**: 
   Introduces a programmatic verification suite designed to validate dataset structure, detect duplicate entries, identify corrupted images, and check diagnosis clinical ranges before model execution. This replaces manual auditing and reduces training failures.

2. **Deterministic Metadata Generation Pipeline**:
   Establishes a metadata generation module that precomputes and centralizes image dimensions, aspect ratios, file sizes, and class distributions. Centralizing these values avoids repeated disk scans during training, reducing I/O latency.

3. **Reproducible PyTorch Data Pipeline**:
   Implements a reproducible, end-to-end data pipeline integrating custom `Dataset` wrapping, stratified data partitioning, torchvision augmentation, and memory-pinned loaders.

4. **Reusable Modular Architecture**:
   Organizes the pipeline into decoupled, single-responsibility modules (`config.py`, `dataset.py`, `transforms.py`, `dataloader.py`) that can be maintained and tested independently.

5. **Complete Data Provenance and Audit Trail**:
   Preserves data history by linking each raw image file directly to its clinical label, verification results, and split metadata, ensuring full reproducibility.

6. **Scalable Multi-Modal Design**:
   Establishes clean directory structures and configuration conventions that easily scale to support additional medical datasets (e.g. IDRiD, DFUC, PIMA) without changing core pipeline logic.
