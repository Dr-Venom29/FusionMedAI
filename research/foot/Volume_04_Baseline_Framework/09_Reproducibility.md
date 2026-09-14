# 09 Reproducibility Verification — Phase 10.4

## Reproducibility Protocol

- **Random Seed**: `42` enforced across Python `random`, `numpy`, PyTorch CPU, and PyTorch CUDA.
- **CuDNN Directives**: `deterministic = True`, `benchmark = False`.
- **Archived Environment**: PyTorch environment, dataset split indices, model state dicts, optimizer state, training history, and CLI parameters stored in `experiments/foot/baseline_resnet50_unweighted/`.
