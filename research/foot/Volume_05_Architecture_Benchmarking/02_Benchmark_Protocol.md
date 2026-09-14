# 02 Benchmark Protocol — Phase 10.5

## Frozen Benchmark Parameters

- **Dataset**: Frozen Phase 10.2 split (8,038 Train / 1,006 Val / 1,006 Test)
- **Source Groups**: 1,770 groups (0% data leakage across splits)
- **Target Classes**: 4 Wagner Classes (`Grade 1`, `Grade 2`, `Grade 3`, `Grade 4`)
- **Input Resolution**: $224 \times 224 \times 3$ RGB
- **Pretrained Weights**: ImageNet pre-trained weights for all 5 candidates
- **Random Seed**: `42` (Enforced across Python, NumPy, PyTorch CPU/CUDA)
- **Batch Size**: `32`
- **Optimizer**: `AdamW` ($\text{lr} = 1\text{e-}4, \text{weight\_decay} = 1\text{e-}4$)
- **Scheduler**: `CosineAnnealingLR` ($T_{\text{max}} = 20, \eta_{\text{min}} = 1\text{e-}6$)
- **Loss Function**: Standard Unweighted CrossEntropy Loss
- **Maximum Epochs**: `20` (Patience = `10` on `val_loss`)
