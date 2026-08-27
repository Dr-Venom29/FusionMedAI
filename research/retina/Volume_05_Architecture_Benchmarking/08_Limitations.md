# Chapter 8: Limitations

## 8.1 Acknowledged Constraints
- **Single Dataset Evaluation**: The models were only benchmarked on the APTOS 2019 dataset. Performance may vary on out-of-distribution datasets (e.g., EyePACS, Messidor).
- **Single Input Resolution**: All architectures were evaluated at 224x224. Larger resolutions may change the relative ranking, particularly for transformer-based models.
- **No Hyperparameter Tuning**: To maintain fairness, generic hyperparameters were used. Models like ViT might have performed better with heavy, architecture-specific tuning or varied learning rates.
- **No External Validation**: The 10% test split is from the same distribution as the training data.
- **ImageNet Initialization**: All models were constrained to weights pre-trained on ImageNet. Pre-training on medical datasets (e.g., RadImageNet) was not evaluated.
- **Post-Hoc Analysis Only**: Calibration, explainability, and uncertainty estimation were intentionally excluded from the benchmark itself and evaluated only after selecting the final backbone.
