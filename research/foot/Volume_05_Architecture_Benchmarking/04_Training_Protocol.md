# 04 Training Protocol — Fresh State Per Model

## Training Runner & Memory Directives

- **Runner Script**: `src/foot/training/benchmark.py`
- **Fresh State Isolation**: Each candidate model receives completely fresh instances of model weights, optimizer state, scheduler state, epoch counters, and metric histories.
- **GPU Memory Release**: Explicit `del model, optimizer, scheduler, trainer`, `gc.collect()`, and `torch.cuda.empty_cache()` executed between model training runs.
- **Checkpoint Directory Isolation**:
  - `experiments/foot/architecture_benchmark/<model_name>/checkpoints/best_model.pt`
  - `experiments/foot/architecture_benchmark/<model_name>/checkpoints/last_model.pt`
  - `experiments/foot/architecture_benchmark/<model_name>/config.json`
