# Chapter 7: Engineering Decisions

This chapter documents the key technical decisions, trade-offs, and software engineering design principles implemented during the dataset preparation phase of the FusionMedAI project.

---

## Technical Decisions, Rationale, and Trade-offs

### 1. Why `pathlib` Instead of `os.path`?
- **Decision**: All paths are constructed and managed using Python's object-oriented `pathlib.Path`.
- **Engineering Rationale**: 
  - `os.path` relies on string concatenation, which is error-prone and platform-dependent (e.g., slash `/` vs. backslash `\`).
  - `pathlib` provides a clean, object-oriented API that resolves path operators portably across Linux, macOS, and Windows environments. This prevents path errors when moving the code from local Windows development environments to Linux-based high-performance computing or cloud GPU environments.
- **Trade-off**: Requires developers to become familiar with object-oriented path handling and conversion (e.g. converting `Path` objects to strings when interfaces require them) rather than using simple string manipulation.

### 2. Why Centralized `config.py`?
- **Decision**: Created the central configuration module (`src/config.py`) to centralize all paths, random seeds, class configurations, and training hyperparameters.
- **Engineering Rationale**:
  - Eliminates path and variable duplication across multiple scripts, implementing the **DRY (Don't Repeat Yourself)** principle.
  - Centralizing paths makes directory restructuring straightforward: if the raw dataset location changes, it only needs to be updated in a single file instead of multiple files.
  - Prevents configuration drift by ensuring that all components use identical configurations.
- **Trade-off**: Couples all scripts to a single configuration module, meaning changes to the config structure can affect multiple operational files.

### 3. Why Metadata Pre-generation Instead of Dynamic Recomputation?
- **Decision**: Precalculate all image properties (width, height, file size, aspect ratio) and save them to CSV and JSON files in `datasets/metadata/` rather than reading raw images dynamically during training.
- **Research Rationale**:
  - Fundus images in the APTOS dataset possess highly variable resolutions (ranging from 474 pixels to 4,288 pixels). Extracting these dimensions on-the-fly inside PyTorch data loaders during training introduces unnecessary computational and I/O overhead.
  - Pre-generating metadata files allows scripts to load the entire dataset index in milliseconds, avoiding CPU bottlenecks and saving disk I/O operations during training epochs.
- **Trade-off**: Metadata becomes stale if raw images change. Therefore, metadata must always be regenerated after dataset modifications.

### 4. Why Verification Before Preprocessing?
- **Decision**: Enforce dataset integrity checks before running preprocessing or resizing.
- **Engineering Rationale**:
  - Medical image preprocessing (resizing, padding, crop mask optimization) is computationally expensive.
  - Running preprocessing on unverified datasets risks wasting hours of processing time before hitting a corrupted file, missing label, or out-of-bounds index. Verification ensures that the data is structurally sound before any processing occurs, following a strict **fail-fast** approach (Shore, 2004).
- **Trade-off**: Adds an initial verification runtime step before preprocessing can begin, increasing pipeline latency on first run.

### 5. Why Separating Raw and Processed Datasets?
- **Decision**: The `datasets/raw/` folder is read-only, and any preprocessed files are written to `datasets/processed/`.
- **Research Rationale**:
  - Raw clinical scans are the ground-truth audit trail (immutable artifacts). Modifying them destroys the ability to trace training data back to the original source.
  - Preserving raw data allows researchers to test different preprocessing and augmentation strategies without needing to re-download the source files.
- **Trade-off**: Increases disk storage requirements since raw and processed images coexist simultaneously.

### 6. Why JSON Plus CSV?
- **Decision**: Use JSON for dataset-level summaries (`dataset_statistics.json`) and CSV for tabular per-image lists (`train_metadata.csv`, `image_statistics.csv`).
- **Engineering Rationale**:
  - **CSV (Tabular Data)**: Highly efficient for scanning, filtering, and joining tables using pandas. PyTorch dataset classes can easily load CSV rows via index lookups.
  - **JSON (Hierarchical Data)**: Supports nested objects and key-value pairs. Well suited for configuration, automation metadata, and experiment logging.
- **Trade-off**: Requires maintaining parser logic for two separate file types.

### 7. Why Modular Scripts Instead of Monolithic Jupyers?
- **Decision**: Implement verification and metadata generation in pure Python scripts (`verify_dataset.py`, `generate_metadata.py`) under `src/data/` rather than in Jupyter Notebooks.
- **Engineering Rationale**:
  - Pure Python scripts can be version-controlled cleanly with Git, without the clutter of large JSON notebook outputs.
  - Scripts are easily executed from the terminal, making them ready for automated CI/CD pipelines, automated testing, and cloud GPU environments.
- **Trade-off**: Lacks the interactive, cell-by-cell visualization interface of Jupyter Notebooks during active code development.

---

## Software Architecture Patterns

- **Configuration Pattern (Single Source of Truth)**: Centralizing configuration constants inside `src/config.py` prevents configuration drift across scripts.
- **Pipeline Pattern**: Directs data ingestion sequentially (Verification -> Metadata -> EDA -> Preprocessing -> Model Training) to enforce data integrity before computation.
- **Factory-like Dataset Construction**: Decouples dataset structure from image transformation operations, allowing the same dataset class to instantiate training, validation, or test batches.

---

## Design Principles Followed

1. **Single Responsibility Principle (SRP)** (Martin, 2003):
   - Each script has one responsibility: `verify_dataset.py` validates integrity, while `generate_metadata.py` handles the construction of summaries.
2. **Separation of Concerns (SoC)**:
   - Configuration, verification logic, and dataset documentation are kept separated, ensuring changes to one do not affect another.
3. **Don't Repeat Yourself (DRY)** (Hunt & Thomas, 1999):
   - Common configuration paths and constants are isolated within `src/config.py` rather than duplicated.
4. **Keep It Simple (KISS)**:
   - Verification and metadata generation remain separate scripts, keeping each component small, testable, and understandable.
5. **Open-Closed Principle (OCP)** (Martin, 2003):
   - New datasets can be added without modifying existing verification logic, only extending dataset-specific configuration files.
6. **Independent Unit Testing**:
   - Every script was verified independently before integration, reducing debugging complexity.

---

## Modularity & Pipeline Dependencies
The following flowchart illustrates the dependency structure of the modules, showing how the centralized configuration feeds all operations:

```
src/config.py
      │
      ├───────────────┐
      ▼               ▼
verify_dataset  generate_metadata
      │               │
      └───────┬───────┘
              ▼
             EDA
              │
              ▼
        split_dataset
              │
              ▼
           Dataset
              │
              ▼
           Training
```
*Figure 7.1: Pipeline dependency diagram showing unidirectional data flow.

---

## References
- Hunt, A., & Thomas, D. (1999). *The Pragmatic Programmer: From Journeyman to Master*. Addison-Wesley.
- Martin, R. C. (2003). *Agile Software Development, Principles, Patterns, and Practices*. Prentice Hall.
- Shore, J. (2004). Fail Fast. *IEEE Software*, 21(5), 21-25.*
