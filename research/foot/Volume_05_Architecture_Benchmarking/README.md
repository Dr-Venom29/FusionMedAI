# Volume V — Foot Ulcer Architecture Benchmarking

## Phase 10.5 — Architecture Benchmarking

This research volume documents the controlled benchmarking of candidate vision architectures for the **Foot Ulcer (DFU) Wagner 4-Class Classification** modality.

The goal of Phase 10.5 is to evaluate five candidate vision architectures under a frozen, reproducible experimental protocol to select the optimal model backbone relative to the ResNet-50 baseline.

---

## Volume Structure

1. **[01 Objectives](file:///d:/FusionMedAI/research/foot/Volume_05_Architecture_Benchmarking/01_Objectives.md)**
2. **[02 Benchmark Protocol](file:///d:/FusionMedAI/research/foot/Volume_05_Architecture_Benchmarking/02_Benchmark_Protocol.md)**
3. **[03 Model Configurations](file:///d:/FusionMedAI/research/foot/Volume_05_Architecture_Benchmarking/03_Model_Configurations.md)**
4. **[04 Training Protocol](file:///d:/FusionMedAI/research/foot/Volume_05_Architecture_Benchmarking/04_Training_Protocol.md)**
5. **[05 Evaluation Protocol](file:///d:/FusionMedAI/research/foot/Volume_05_Architecture_Benchmarking/05_Evaluation_Protocol.md)**
6. **[06 Results](file:///d:/FusionMedAI/research/foot/Volume_05_Architecture_Benchmarking/06_Results.md)**
7. **[07 Model Comparison](file:///d:/FusionMedAI/research/foot/Volume_05_Architecture_Benchmarking/07_Model_Comparison.md)**
8. **[08 Selection](file:///d:/FusionMedAI/research/foot/Volume_05_Architecture_Benchmarking/08_Selection.md)**

---

## Candidate Architectures

The benchmarking suite evaluates five candidate families:

1. **EfficientNet-B0** (~4.0M parameters)
2. **EfficientNet-B3** (~10.7M parameters)
3. **ConvNeXt-Tiny** (~27.8M parameters)
4. **Swin-Tiny** (~27.5M parameters)
5. **ViT-B/16** (~85.8M parameters)

---

## Benchmark Directives

- **Reference Baseline**: ResNet-50 (Phase 10.4 Macro F1 = `0.6339`, Accuracy = `63.72%`)
- **Dataset Partition**: Frozen Phase 10.2 split (8,038 Train / 1,006 Val / 1,006 Test)
- **Input Resolution**: $224 \times 224 \times 3$ RGB
- **Optimizer & Loss**: AdamW ($\text{lr} = 1\text{e-}4$, $\text{weight\_decay} = 1\text{e-}4$), Unweighted CrossEntropy
- **Checkpoint Policy**: Minimum validation loss (`val_loss`) for early stopping and model selection
- **Primary Metric**: **Macro F1-Score** on the frozen test partition
