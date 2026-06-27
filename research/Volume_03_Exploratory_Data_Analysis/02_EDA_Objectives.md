# Chapter 2: EDA Objectives

## Core Research Objectives
The core research objectives of this Exploratory Data Analysis phase are defined below:

1. **Perform Quantitative and Visual Profiling**: Establish a complete descriptive understanding of the clinical fundus dataset before applying any preprocessing, data resizing, or normalization transforms.
2. **Evaluate Clinical Class Imbalance**: Quantify the distribution of Diabetic Retinopathy severity stages in the training cohort to identify the clinical risks associated with class-wise sensitivity and recommend appropriate mathematical weightings.
3. **Conduct Dataset-wide Color Distribution Profiling**: Measure the global and class-wise RGB channel distributions to evaluate whether disease progression causes illumination shifts, and establish a baseline for normalizations.
4. **Identify Spatial and Aspect Ratio Variances**: Profile the range of image resolutions, aspect ratios, and spatial scales to determine scaling and cropping methods that prevent shape distortion.
5. **Develop and Evaluate a Continuous Image Quality Metric ($Q$)**: Integrate spatial resolution, brightness, and sharpness measurements to characterize clinical image usability and rank image suitability.
6. **Detect Data Leakage via Exact Duplicate Image Detection**: Implement a pixel-exact hashing pipeline to audit the dataset for hidden duplicate records, preventing train-validation leakage.
7. **Establish Dataset-level Baseline Statistics**: Formulate baseline metrics that will guide preprocessing, normalization, augmentation, and loss-function selection during subsequent model development.

---

## Technical and Engineering Goals
To support these research objectives, the EDA module must achieve the following engineering goals:

1. **High-Throughput Parallel Processing**: Implement a multi-threaded image scanning pipeline utilizing `ThreadPoolExecutor` and in-memory scaling to reduce disk I/O bottlenecks and process the entire dataset in under $90$ seconds.
2. **Deterministic Reproducibility Controls**: Establish a strict seed control (`SEED = 42`) and a dataset SHA-256 fingerprinting protocol to ensure audit trails remain completely reproducible.
3. **Read-Only Verification Suite**: Perform all metadata extractions under read-only constraints, ensuring no raw image arrays are modified or resized on disk.
4. **Structured Outlier Mapping**: Generate CSV tables containing statistically identified dark, bright, blurry, large, and small images for manual clinical quality assessment.
5. **Unified Reporting and Exporters**: Implement automated tools to compile vector PDF reports, Markdown summaries, and execution manifests containing environment configurations.
6. **Construct Preprocessing Recommendations**: Export candidate configurations for image size, color normalization, augmentations, and loss functions as structured JSON recommendations to be benchmarked during the model development and training phase.
