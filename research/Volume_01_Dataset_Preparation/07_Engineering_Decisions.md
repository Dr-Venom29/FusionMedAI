# Chapter 7: Engineering Decisions

This chapter documents the key technical decisions, trade-offs, and software engineering design principles implemented during the dataset preparation phase of the FusionMedAI project.

---

## Technical Decisions and Rationale

### 1. Why `pathlib` Instead of `os.path`?
- **Decision**: All paths are constructed and managed using Python's object-oriented `pathlib.Path`.
- **Engineering Rationale**: 
  - `os.path` relies on string concatenation, which is error-prone and platform-dependent (e.g., slash `/` vs. backslash `\`).
  - `pathlib` provides a clean, object-oriented API that resolves path operators portably across Linux, macOS, and Windows environments. This prevents path errors when moving the code from local Windows development environments to Linux GPU clusters.

### 2. Why Centralized `config.py`?
- **Decision**: Created [src/config.py](file:///d:/FusionMedAI/src/config.py) to centralize all paths, random seeds, class configurations, and training hyperparameters.
- **Engineering Rationale**:
  - Eliminates path and variable duplication across multiple scripts (`verify_dataset.py`, `generate_metadata.py`, etc.).
  - Centralizing paths makes directory restructuring straightforward: if the raw dataset location changes, it only needs to be updated in a single file instead of multiple files.
  - Prevents configuration drift by ensuring that all components (data, training, and evaluation) use identical configurations.

### 3. Why Metadata Pre-generation Instead of Dynamic Recomputation?
- **Decision**: Precalculate all image properties (width, height, file size, aspect ratio) and save them to CSV and JSON files in `datasets/metadata/` rather than reading raw images dynamically during training.
- **Research Rationale**:
  - Fundus images in the APTOS dataset possess highly variable resolutions (ranging from 474 pixels to 4,288 pixels). Extracting these dimensions on-the-fly inside PyTorch data loaders during training is computationally expensive.
  - Pre-generating metadata files allows scripts to load the entire dataset index in milliseconds, avoiding CPU bottlenecks and saving disk I/O operations during training epochs.

### 4. Why Verification Before Preprocessing?
- **Decision**: Enforce dataset integrity checks before running preprocessing or resizing.
- **Engineering Rationale**:
  - Medical image preprocessing (resizing, padding, crop mask optimization) is computationally expensive.
  - Running preprocessing on unverified datasets risks wasting hours of processing time before hitting a corrupted file, missing label, or out-of-bounds index. Verification ensures that the data is structurally sound before any processing occurs.

### 5. Why Separating Raw and Processed Datasets?
- **Decision**: The `datasets/raw/` folder is read-only, and any preprocessed files are written to `datasets/processed/`.
- **Research Rationale**:
  - Raw clinical scans are the ground-truth audit trail. Modifying them destroys the ability to trace training data back to the original source.
  - Preserving raw data allows researchers to test different preprocessing and augmentation strategies (e.g., resizing to 224px vs. 512px) without needing to re-download the source files.

### 6. Why JSON Plus CSV?
- **Decision**: Use JSON for dataset-level summaries (`dataset_statistics.json`) and CSV for tabular per-image lists (`train_metadata.csv`, `image_statistics.csv`).
- **Engineering Rationale**:
  - **JSON** is highly structured, supporting nested objects and key-value pairs. It is ideal for storing overall metadata parameters (such as mean dimensions, class names, and dataset lists) that are easily read by scripts and configuration files.
  - **CSV** is tabular, making it highly efficient for scanning, filtering, and joining tables using pandas. PyTorch dataset classes can easily load CSV rows via index lookups.

### 7. Why Modular Scripts Instead of Monolithic Jupyers?
- **Decision**: Implement verification and metadata generation in pure Python scripts (`verify_dataset.py`, `generate_metadata.py`) under `src/data/` rather than in Jupyter Notebooks.
- **Engineering Rationale**:
  - Pure Python scripts can be version-controlled cleanly with Git, without the clutter of large JSON notebook outputs.
  - Scripts are easily executed from the terminal, making them ready for CI/CD pipelines, automated testing, and remote GPU cluster runs.

---

## Design Principles Followed

1. **Single Responsibility Principle (SRP)**:
   - Each script has one responsibility: `verify_dataset.py` validates integrity, while `generate_metadata.py` handles the construction of summaries.

2. **Separation of Concerns (SoC)**:
   - Configuration (`src/config.py`), verification logic (`src/data/verify_dataset.py`), and dataset documentation are kept separated, ensuring changes to one do not affect another.

3. **Modularity**:
   - The codebase is structured with clear imports: data scripts import directly from the centralized `src/config.py` module, establishing a clean dependency graph.

4. **Reproducibility**:
   - Every file path, seed, and data layout is fixed, ensuring that another researcher starting with the raw files will get identical verification outputs.

5. **Maintainability**:
   - Readable formatting and descriptive naming make it easy to extend the scripts to support other datasets (such as IDRiD) in future volumes.
