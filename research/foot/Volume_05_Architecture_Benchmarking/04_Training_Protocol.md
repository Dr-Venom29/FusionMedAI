# 04 Training Protocol — Fresh State & Artifact Isolation

## 10.5.9 Training Framework & Memory Directives

- **Single Training Runner**: Executed via `src/foot/training/benchmark.py` across all candidate architectures.
- **Fresh State Isolation**: Each candidate model receives fresh, un-shared instances of model parameters, optimizer state (`AdamW`), scheduler state (`CosineAnnealingLR`), and training history data structures.
- **Pre-Training Logits Contract**: Prior to training, each instantiated model passes explicit format verification (`isinstance(output, dict)` and `logits.shape == (2, 4)`).
- **GPU Memory Cleanup**: Memory cleanup operations (`del model, optimizer, scheduler, trainer`, `gc.collect()`, `torch.cuda.empty_cache()`) executed sequentially between benchmarking candidate models to prevent inter-model state contamination.

---

## 10.5.10 Checkpoint & Artifact Directory Isolation

Every candidate architecture maintains isolated experiment outputs under `experiments/foot/architecture_benchmark/<model_name>/`:

```
experiments/foot/architecture_benchmark/
├── benchmark_config.json
├── model_profiling.csv
├── benchmark_results.csv
├── benchmark_results.json
├── benchmark_report.md
├── metadata/
│   ├── efficientnet_b0_confusion_matrix.png
│   ├── efficientnet_b3_confusion_matrix.png
│   ├── convnext_tiny_confusion_matrix.png
│   ├── swin_tiny_confusion_matrix.png
│   └── vit_b16_confusion_matrix.png
├── efficientnet_b0/
│   ├── checkpoints/
│   │   ├── best_model.pt
│   │   └── last_model.pt
│   ├── config.json
│   ├── validation_metrics.json
│   ├── test_evaluation.json
│   ├── training_history.csv
│   ├── confusion_matrix.png
│   └── error_analysis.csv
├── efficientnet_b3/
│   └── checkpoints/ ...
├── convnext_tiny/
│   └── checkpoints/ ...
├── swin_tiny/
│   └── checkpoints/ ...
└── vit_b16/
    └── checkpoints/ ...
```

---

## 10.5.11 Self-Describing Checkpoint State

Checkpoints saved by `CheckpointManager` store full self-describing metadata in the `.pt` dictionary:
- `epoch`: Epoch number of checkpoint save
- `model_state_dict`: Model parameters dictionary
- `optimizer_state_dict`: Optimizer state dictionary
- `scheduler_state_dict`: Scheduler state dictionary
- `best_val_loss`: Minimum validation loss achieved (`val_loss`)
- `best_val_macro_f1`: Validation Macro F1 at best validation-loss epoch
- `checkpoint_metric`: `"val_loss"`
- `config`: BaselineConfig parameter dictionary
- `seed`: `42`
