# 9. Reproducibility

To ensure strict scientific rigor, the calibration process is entirely reproducible.

## Artifacts Generated
For every run of the calibration pipeline, an isolated experiment folder is created under `experiments/calibration/vXXX_temperature_scaling/` containing:
- `manifest.json`: Records the dataset version, model architecture, checkpoint name, PyTorch/CUDA versions, random seed, learned temperature, and the specific git commit hash.
- `temperature_scaling.pt`: The state dictionary of the wrapped `ModelWithTemperature`.
- `calibration_state.pt`: Contains the optimal temperature and associated metrics.
- `tensorboard/`: Event logs tracking metrics before and after optimization.
- `calibration_metrics.json` and `.csv`: Detailed numerical results.

The CLI ensures deterministic behavior by setting standard random seeds across numpy and PyTorch.
