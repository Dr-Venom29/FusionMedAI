# 04 Training Framework — Foot Baseline

## Modular Infrastructure

The training framework is implemented in `src/foot/training/`:
- **`config.py`**: Centralized configuration management and Kaggle read-only dataset compatibility (`FUSIONMEDAI_FOOT_DATASET_ROOT`).
- **`losses.py`**: Configurable loss factory supporting both unweighted and sqrt inverse-frequency weighted CrossEntropy.
- **`metrics.py`**: Automated metric computation (`compute_baseline_metrics`).
- **`checkpoint.py`**: Reproducible checkpoint state saving (`best_model.pt` and `last_model.pt`).
- **`trainer.py`**: `FootBaselineTrainer` with device ownership safety and early stopping.
- **`train_baseline.py`**: Standalone CLI runner for local and Kaggle execution.
