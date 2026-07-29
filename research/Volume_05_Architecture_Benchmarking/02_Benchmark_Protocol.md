# Chapter 2: Benchmark Protocol

## 2.1 The Need for Reproducibility
To ensure a scientifically fair comparison, the benchmark design strictly freezes all training variables. Every model architecture is subjected to the exact same conditions.

## 2.2 Experimental Setup
- **Dataset**: APTOS 2019 Blindness Detection
- **Train/Val/Test Split**: 70/15/15 (Stratified, identical split for all runs)
- **Image Size**: 224x224 (Fixed across all architectures, including ViT)
- **Batch Size**: 32
- **Optimizer**: AdamW
- **Scheduler**: CosineAnnealingLR
- **Learning Rate**: 1e-4
- **Maximum Epochs**: 50
- **Early Stopping Patience**: 10
- **Hardware Profile Logging**: Enabled (Params, FLOPs, MACs, Peak VRAM)
- **Random Seed**: 42 (Deterministic execution)

## 2.3 Fairness Constraints
To explicitly isolate model architecture as the only independent variable, the following constraints are rigorously enforced:
* Same train/validation/test split
* Same augmentation policy
* Same preprocessing
* Same random seed
* Same optimizer
* Same scheduler
* Same early stopping
* Same evaluation metrics
