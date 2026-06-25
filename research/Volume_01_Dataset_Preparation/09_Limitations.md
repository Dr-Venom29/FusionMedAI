# Chapter 9: Limitations

This chapter documents the limitations of the current dataset preparation phase and explains why certain steps are deferred to later stages of the project.

---

## Documented Limitations

### 1. Only APTOS 2019 Prepared
- *Limitation*: Only the APTOS 2019 dataset has been verified and prepared. Other datasets (such as IDRiD for fundus photography, DFUC for foot ulcers, and PIMA for clinical tabular data) have not yet been integrated.
- *Reason for Deferral*: Focusing on a single dataset allows us to build, test, and refine the verification and metadata pipelines before extending them to other modalities.

### 2. No Image Preprocessing
- *Limitation*: No image resizing, cropping, aspect ratio padding, or normalization has been applied to the raw fundus images.
- *Reason for Deferral*: Preprocessing parameters (such as the target image resolution or normalization means/stds) must be decided based on findings from the Exploratory Data Analysis (EDA) in Step 2.

### 3. No Data Augmentation
- *Limitation*: No augmentations (e.g., horizontal/vertical flips, rotations, color jitter, CLAHE) have been configured.
- *Reason for Deferral*: Augmentation strategies are closely tied to model training and must be configured inside the training script (`train.py`) to run on the GPU, avoiding unnecessary disk overhead.

### 4. No Image Quality Enhancement
- *Limitation*: No fundus image quality enhancement (such as Ben Graham's processing or circular cropping to remove black borders) has been applied.
- *Reason for Deferral*: Fundus preprocessing is a specialized task. Implementing it now would mix dataset ingestion with image preprocessing, violating the Separation of Concerns principle.

### 5. No Train/Validation/Test Splitting
- *Limitation*: A stratified split (e.g., 80% train, 10% validation, 10% test) has not yet been created.
- *Reason for Deferral*: Splitting requires loading the class distribution to perform stratified sampling. This step is deferred to the preprocessing phase in Step 3 to ensure splits are generated and recorded programmatically.

### 6. No Model Training or Evaluation
- *Limitation*: No neural networks have been developed, trained, or evaluated.
- *Reason for Deferral*: Training requires a verified dataset, stratified splits, and defined preprocessing steps. Developing models before establishing these components risks training on corrupted data or introducing data leakage.

### 7. Verification Confirms Structural Integrity, Not Clinical Correctness
- *Limitation*: The verification script checks structural parameters (e.g., checking if the label is an integer between 0 and 4). It cannot verify the clinical accuracy of the labels assigned by human graders.
- *Reason for Deferral*: Clinical validation is outside the scope of automated data pipelines. We assume the annotations in the public dataset are correct and focus on ensuring their structural integrity.
