# Chapter 6: Metadata Generation

This chapter details the purpose, structure, columns, and downstream reuse of the generated metadata files.

---

## Why Metadata Matters
In deep learning pipelines, especially when dealing with high-resolution image data, disk and memory I/O can easily introduce unnecessary I/O overhead during preprocessing and training. Traversing thousands of directories to fetch image dimensions, file sizes, or labels during preprocessing or training is computationally expensive. Generating a centralized metadata directory solves this bottleneck.

Metadata acts as a lightweight, structured representation of the dataset. Instead of reading thousands of large images to calculate distributions, check dimensions, or verify labels, scripts can load a single CSV or JSON file in milliseconds, significantly reducing CPU memory overhead.

The relationship between the raw data and the generated metadata files is illustrated below:

```
Raw CSV Indices (train.csv / test.csv)
                │
                ├───► Metadata Generator (generate_metadata.py)
                │                  │
Raw Image Files ┘                  ▼
                           metadata/ directory
                                   │
      ┌────────────────┬───────────┴────┬─────────────────┐
      ▼                ▼                ▼                 ▼
train_metadata.csv image_statistics.csv class_distribution.csv dataset_statistics.json
```
*Figure 6.1: Relationship between raw inputs and generated metadata files.*

---

## Metadata Philosophy
Metadata generation follows a "compute once, reuse many times" philosophy. Rather than repeatedly extracting image properties during preprocessing or model training, the pipeline computes descriptive statistics once and stores them in lightweight structured files. This reduces redundant computation, improves consistency across experiments, and simplifies downstream analysis.

### Data Provenance
In medical AI systems, tracking the origin, processing path, and characteristics of clinical data is vital. The metadata generation stage establishes **data provenance** by linking each raw image file directly to its clinical label, measured spatial attributes, and verification status. This ensures that every tensor fed into the training loop can be audited back to its clinical source.

---

## Metadata Files Summary
The generated metadata files and their key attributes are summarized in the table below:

| Metadata File | Purpose | Key Columns / Fields | Downstream Target |
| :--- | :--- | :--- | :--- |
| **`train_metadata.csv`** | Master training index | `id_code`, `filename`, `label`, `width`, `height`, `aspect_ratio`, `filesize_kb` | Custom PyTorch Dataset class |
| **`image_statistics.csv`** | Individual file statistics | `id_code`, `filename`, `diagnosis`, `width`, `height`, `channels`, `aspect_ratio`, `filesize_kb` | EDA Notebooks, Preprocessing |
| **`class_distribution.csv`** | Imbalance summary | `Label`, `Severity`, `Count` | Loss weights calculation |
| **`quality_statistics.csv`** | Quality verification logging | `Metric`, `Value` (e.g. RGB Images, Corrupted Images) | CI/CD Auditing Pipeline |
| **`dataset_statistics.json`**| Global dataset summary | `dataset`, `training_images`, `testing_images`, `average_width`, `average_height`, sizes | Logging, Experiment Configuration |

Because metadata are generated deterministically from verified raw data, multiple researchers can reproduce identical metadata files from the same dataset. This improves experiment reproducibility and auditability.

---

## Metadata Lifecycle
The following flowchart displays the complete lifecycle of metadata from initial clinical acquisition to final research archiving:

```
Raw Images
     │
     ▼
Verification
     │
     ▼
  Metadata
     │
     ▼
    EDA
     │
     ▼
   Split
     │
     ▼
  Dataset
     │
     ▼
  Training
     │
     ▼
 Evaluation
     │
     ▼
Experiments
     │
     ▼
  Archive
```
*Figure 6.2: Complete metadata lifecycle stages.*

---

## Detailed Metadata Structure

### 1. `train_metadata.csv`
- **Purpose**: Serves as the master index for the training split, uniting tabular patient labels with verified image attributes.
- **Columns**:
  - `id_code`: The unique patient identifier string (matches original clinical index).
  - `filename`: The exact filename in the directory (e.g., `000c1434d8d7.png`).
  - `label`: The expert-graded severity score (0, 1, 2, 3, or 4).
  - `width` & `height`: Raw dimensions in pixels.
  - `aspect_ratio`: Calculated as width divided by height (rounded to 2 decimal places).
  - `filesize_kb`: The disk size in kilobytes (rounded to the nearest integer).

### 2. `image_statistics.csv`
- **Purpose**: Records individual statistics for each image to assist in Exploratory Data Analysis (EDA) and batch normalization planning.
- **Columns**:
  - `id_code`, `filename`, `diagnosis`.
  - `width`, `height`, `channels` (strictly 3 for RGB).
  - `aspect_ratio`, `filesize_kb`.
- **Research Value**: The high variability in fundus image dimensions (ranging from 474 pixels to 4,288 pixels) requires metadata-driven preprocessing. This CSV helps identify size groupings and outlier dimensions, guiding resizing, augmentation, and padding strategies.

### 3. `class_distribution.csv`
- **Purpose**: Summarizes the count and proportion of each diabetic retinopathy severity stage.
- **Why Imbalance Analysis Matters**: Medical datasets are almost always highly imbalanced. The generated distribution shows that Class 0 (No DR) accounts for **49.29%** of the dataset, while Class 3 (Severe) accounts for only **5.27%**. Downstream training scripts read this CSV to compute class weights for the loss function (e.g., `WeightedCrossEntropyLoss`) or configure oversampling/undersampling parameters.

### 4. `dataset_statistics.json`
- **Purpose**: Provides a machine-readable summary of the entire dataset. It is well suited for storing overall metadata parameters (such as mean dimensions, class names, and dataset lists) that are easily read by scripts and configuration files, helping prevent configuration drift.

---

## Metadata Reuse and Data Flow
The pre-generated metadata is reused across multiple stages of the FusionMedAI pipeline:

```
                          datasets/metadata/
                                   │
      ┌────────────┬───────────────┼───────────────┬────────────┐
      ▼            ▼               ▼               ▼            ▼
EDA & Quality   PyTorch      Loss Function   Preprocessing    CI/CD
  Analysis      Dataset        Weighting    Recommendation   Auditing
```
*Figure 6.3: Downstream modules consuming generated metadata.*

1. **Exploratory Data & Quality Analysis**:
   - EDA pipelines load `image_statistics.csv` and `class_distribution.csv` directly to plot distributions, verify aspect ratios, and analyze outlier dimensions, reducing computational overhead. This is consumed by quality modules to compute continuous Quality Scores ($Q$) and flag dark, bright, or blurry outliers.
2. **PyTorch Dataset Class**:
   - The custom dataset class loads `train_metadata.csv` to read target labels and image filenames. This avoids performing expensive directory scans or reading image files on-the-fly during training.
3. **Training & Loss Optimization**:
   - Training scripts load `class_distribution.csv` to compute class-weight arrays, balancing the training loss to prevent the network from ignoring minority classes (e.g., Severe DR).
4. **Preprocessing Recommendation Engine**:
   - Analyzes the statistical distribution of spatial dimensions and color channel statistics to establish candidate resizing resolutions, normalizations, and augmentation parameters.
5. **Experiments Log & Reporting**:
   - Experiment managers (like MLflow or TensorBoard) read `dataset_statistics.json` to log hyperparameters and dataset attributes, establishing a solid audit trail for research publication.

---

## Engineering Decisions

- **Why JSON and CSV Formats?**
  We choose the storage format based on the structure of the data and its downstream consumption profile:
  - **CSV (Tabular Data)**: Tabular data (like image sizes and labels) is stored as CSVs, which are directly readable by Pandas during EDA and PyTorch datasets during training.
  - **JSON (Hierarchical Data)**: Config-like global parameters (like mean sizes, class labels, and dataset statistics) are saved as JSON, facilitating quick integration with automated training modules and experiment logs.
- **Why Reusable Reports and Storage Efficiency?**
  Generating reusable reports ensures that dataset verification only needs to be run **once**. Repeatedly opening 3,662 high-resolution PNG images during every preprocessing stage would introduce unnecessary I/O. Instead, loading the `train_metadata.csv` requires only a few hundred KB of memory, which significantly reduces disk I/O latency.
- **No Manual Modifications**
  Generated metadata are treated as derived artifacts. They should never be manually edited, because doing so would break consistency with the verified raw dataset. Instead, metadata should always be regenerated from the source data using the metadata generation pipeline.
- **Dataset-Independent Scalability**
  The metadata generation framework is dataset-independent. Future datasets (such as IDRiD, DFUC, or PIMA) can generate their own metadata using the same pipeline, supporting the extensibility of the overall FusionMedAI project.
- **Influence on Preprocessing Decisions**
  Metadata are not generated solely for documentation; they directly influence downstream pipeline decisions, such as selecting the resizing strategy (maintaining aspect ratio vs. cropping), setting augmentation bounds, defining class balancing weight matrices, and selecting batching dimensions.
