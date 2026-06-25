# Chapter 8: Reproducibility

This chapter outlines the strategies and step-by-step instructions implemented in the FusionMedAI project to guarantee reproducibility across different computing environments.

---

## Why Reproducibility Matters in AI Research
Reproducibility is a cornerstone of scientific research (Gundersen & Kjensmo, 2018). In deep learning and computer vision, studies are frequently plagued by "reproducibility crises" where models fail to achieve published accuracies when trained on other machines. This is often caused by:
- Unrecorded dataset modifications.
- Random initialization drift (lack of fixed seeds).
- Undocumented preprocessing steps.
- Hardcoded directory paths that do not port cleanly to other systems.

Ensuring absolute reproducibility is critical for medical AI to verify clinical claims, audit model predictions, and establish trust with healthcare providers.

---

## Reproducibility Framework

### 1. Fixed Random Seed
- The project configures a global random seed (`SEED = 42`) in [src/config.py](file:///d:/FusionMedAI/src/config.py). 
- In the verification script, setting the random seed ensures that the 100 images sampled for verifying image attributes are identical across every run. This ensures that any warnings or checks can be reproduced exactly.

### 2. Centralized Configuration
- All paths, class definitions, and hyperparameters are declared in [src/config.py](file:///d:/FusionMedAI/src/config.py). This prevents developers from introducing hardcoded paths or parameters in individual scripts, keeping the configuration consistent.

### 3. Immutable Raw Dataset
- The `datasets/raw/` folder is treated as read-only. Scripts are blocked from writing to or modifying files in this directory. If a file is corrupted, it is logged in the metadata directory, keeping the raw dataset unmodified.

### 4. Automated Verification
- Running `verify_dataset.py` validates the dataset integrity and saves the results. This automated step replaces manual ad-hoc checks, providing an auditable proof of dataset health.

### 5. Automated Metadata Generation
- `generate_metadata.py` programmatically creates the metadata files (`train_metadata.csv`, `dataset_statistics.json`, etc.). By automating metadata creation, the dataset parameters remain consistent between runs.

### 6. Git Strategy
- Large binary files (such as raw and processed images) are excluded from version control using `.gitignore`. However, configuration scripts, verification modules, and metadata summaries are committed, allowing researchers to track changes to the dataset's characteristics.

### 7. Consistent Folder Structure
- Enforcing a standardized directory structure (`raw/`, `interim/`, `processed/`, `metadata/`) ensures that scripts can find and write files to the correct locations regardless of the host machine.

---

## Step-by-Step Reproduction Guide

To reproduce the dataset preparation phase from scratch on a new machine, follow these steps:

### Step 1: Clone the Repository
Clone the codebase to your local machine:
```bash
git clone https://github.com/username/FusionMedAI.git
cd FusionMedAI
```

### Step 2: Set Up the Python Environment
Create a virtual environment and install the dependencies listed in `requirements.txt`:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Set Up the Directory Structure
Create the required directory layout (if not already present):
```bash
mkdir -p datasets/raw/aptos2019
mkdir -p datasets/interim
mkdir -p datasets/processed
mkdir -p datasets/metadata
```

### Step 4: Download and Place the Dataset
1. Download the APTOS 2019 Blindness Detection dataset from Kaggle:
   [https://www.kaggle.com/competitions/aptos2019-blindness-detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection)
2. Extract the downloaded files directly into `datasets/raw/aptos2019/`, ensuring it contains:
   - `train_images/`
   - `test_images/`
   - `train.csv`
   - `test.csv`
   - `sample_submission.csv`

### Step 5: Execute Dataset Verification
Run the verification script to audit the dataset and check for errors:
```bash
python src/data/verify_dataset.py
```
This script will verify files, check image modes, verify labels, and generate the following reports:
- `datasets/metadata/verification_report.json`
- `datasets/metadata/verification.log`
- `datasets/metadata/image_sizes.csv`
- `datasets/metadata/missing_images.csv`
- `datasets/metadata/corrupted_images.csv`
- `datasets/metadata/duplicate_ids.csv`
- `datasets/metadata/missing_test_images.csv`
- `datasets/metadata/duplicate_test_ids.csv`

### Step 6: Generate Dataset Metadata
Run the metadata generation script to create the files required for training:
```bash
python src/data/generate_metadata.py
```
This script will produce:
- `datasets/metadata/train_metadata.csv`
- `datasets/metadata/class_distribution.csv`
- `datasets/metadata/dataset_statistics.json`
- `datasets/metadata/image_statistics.csv`
- `datasets/metadata/quality_statistics.csv`

Following these steps will produce identical files, checksums, and verification metrics, confirming the reproducibility of Step 1.

---

## References
- Gundersen, O. E., & Kjensmo, S. (2018). State of the art: Reproducibility in artificial intelligence. *AAAI Conference on Artificial Intelligence*, 32(1), 1644-1651.
