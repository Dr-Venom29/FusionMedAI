# Volume IV — Foot Ulcer Baseline Framework

## Phase 10.4 — Baseline Model Development

This research volume documents the development, training, evaluation, and verification of the baseline classification model for the FusionMedAI Foot Ulcer module.

The purpose of Phase 10.4 is to establish a reproducible reference model using the frozen Foot Ulcer dataset and validated data pipeline. The resulting baseline provides the reference point for the controlled architecture comparison conducted in Phase 10.5.

---

## Volume Structure

1. **[01 Objectives](file:///d:/FusionMedAI/research/foot/Volume_04_Baseline_Framework/01_Objectives.md)**
2. **[02 Experimental Contract](file:///d:/FusionMedAI/research/foot/Volume_04_Baseline_Framework/02_Experimental_Contract.md)**
3. **[03 Baseline Architecture](file:///d:/FusionMedAI/research/foot/Volume_04_Baseline_Framework/03_Baseline_Architecture.md)**
4. **[04 Training Framework](file:///d:/FusionMedAI/research/foot/Volume_04_Baseline_Framework/04_Training_Framework.md)**
5. **[05 Training Protocol](file:///d:/FusionMedAI/research/foot/Volume_04_Baseline_Framework/05_Training_Protocol.md)**
6. **[06 Evaluation Protocol](file:///d:/FusionMedAI/research/foot/Volume_04_Baseline_Framework/06_Evaluation_Protocol.md)**
7. **[07 Baseline Results](file:///d:/FusionMedAI/research/foot/Volume_04_Baseline_Framework/07_Baseline_Results.md)**
8. **[08 Error Analysis](file:///d:/FusionMedAI/research/foot/Volume_04_Baseline_Framework/08_Error_Analysis.md)**
9. **[09 Reproducibility](file:///d:/FusionMedAI/research/foot/Volume_04_Baseline_Framework/09_Reproducibility.md)**
10. **[10 Acceptance](file:///d:/FusionMedAI/research/foot/Volume_04_Baseline_Framework/10_Acceptance.md)**

---

## Scope

Phase 10.4 covers:

1. Baseline experimental definition
2. Dataset and split verification
3. Baseline architecture selection
4. Model implementation
5. Training configuration
6. Training and validation
7. Checkpoint management
8. Frozen test-set evaluation
9. Error analysis
10. Reproducibility verification
11. Baseline acceptance

The following are intentionally outside the scope of this phase:

- Architecture comparison
- Probability calibration
- Predictive uncertainty estimation
- Explainability analysis
- Multimodal fusion
- Clinical risk aggregation

These are addressed in subsequent research phases.

---

## Dataset

The baseline uses the canonical Foot Ulcer modeling population established during Phases 10.1 and 10.2.

### Canonical Population

- Canonical images: **10,050**
- Source-image groups: **1,770**
- Classes: **4**
- Image format: JPEG
- Image resolution: **224 × 224**
- Image channels: RGB

### Final Modeling Splits

| Split | Source Groups | Images |
| :--- | ---: | ---: |
| Train | 1,412 | 8,038 |
| Validation | 177 | 1,006 |
| Test | 181 | 1,006 |
| **Total** | **1,770** | **10,050** |

The final splits were generated using deterministic source-image group-stratified splitting with seed `42`.

No source-image group is shared between the final training, validation, and test partitions.

Patient identifiers are not available in the distributed dataset; therefore, patient-level separation cannot be independently established.

---

## Baseline Model

The baseline architecture is **ResNet-50** with pretrained weights.

The model performs four-class classification:

```text
Input:  224 × 224 RGB image
        ↓
     ResNet-50
        ↓
 4-class classification head
        ↓
Grade 1 / Grade 2 / Grade 3 / Grade 4
```

The baseline experiment uses an unweighted cross-entropy objective.

Class weighting is not applied in the primary baseline because the canonical population has a relatively low class imbalance ratio of 1.18 : 1.

### Data Processing

The baseline uses the validated Foot Ulcer data pipeline established in Phase 10.2.

- **Training**: Resize to 224 × 224, Random rotation: ±15°, Horizontal flip: p=0.5, Color jitter, Observed dataset normalization.
- **Validation and Test**: Resize to 224 × 224, Observed dataset normalization. No stochastic augmentation.

Observed normalization statistics:
- **Mean**: `[0.4937, 0.3630, 0.3272]`
- **Std**: `[0.1745, 0.1632, 0.1551]`

---

## Experimental Configuration

The baseline experiment was executed with:

| Parameter | Value |
| :--- | :--- |
| Architecture | ResNet-50 |
| Pretraining | ImageNet pretrained |
| Number of classes | 4 |
| Batch size | 32 |
| Workers | 4 |
| Seed | 42 |
| Checkpoint criterion | Minimum Validation Loss (`val_loss`) |
| Primary metric | Macro F1-Score |
| Early stopping | Enabled (Patience = 10 on `val_loss`) |

> [!NOTE]
> **Methodological Policy**: Model checkpoint selection and early stopping during training are governed by minimum validation loss (`val_loss`), while Macro F1-score serves as the primary reporting metric for model benchmarking on the frozen test partition.

The complete experiment configuration is archived with the experiment artifacts.

---

## Training

The baseline model was trained for a maximum of 20 epochs.

Early stopping was triggered at epoch 11 after the configured non-improvement interval.

The best checkpoint according to validation loss was obtained at:

- **Epoch**: 1
- **Validation Loss**: 0.9264
- **Validation Macro F1**: 0.6689
- **Validation Accuracy**: 0.6730

The training history indicates rapid reduction of training loss while validation loss remained substantially higher, providing evidence of early overfitting under the baseline configuration.

This observation is retained as part of the baseline characterization and is not used to retrospectively modify the experiment.

---

## Frozen Test Evaluation

The best validation-loss checkpoint was evaluated once against the frozen test partition.

### Overall Results

| Metric | Result |
| :--- | :---: |
| **Macro F1 (Primary Metric)** | **0.6339** |
| **Accuracy** | 0.6372 |
| **Balanced Accuracy** | 0.6391 |
| **Weighted F1** | 0.6319 |
| **Macro ROC-AUC** | 0.8423 |
| **Test Loss** | 1.0548 |

Macro F1 is the primary reporting metric for the baseline.

The resulting **0.6339 Macro F1** is retained as the baseline reference for subsequent Foot Ulcer architecture benchmarking.

### Class-Wise Results

| Class | Precision | Recall | F1 | Support |
| :--- | :---: | :---: | :---: | ---: |
| **Grade 1** | 0.6386 | 0.5443 | 0.5877 | 237 |
| **Grade 2** | 0.7020 | 0.5650 | 0.6261 | 246 |
| **Grade 3** | 0.5684 | 0.5786 | 0.5735 | 280 |
| **Grade 4** | 0.6573 | 0.8683 | 0.7482 | 243 |

Grade 4 produced the highest class-wise F1 score and recall.

Grade 2 and Grade 3 remain difficult classes, consistent with the visual overlap identified during Phase 10.3.

### Confusion Matrix

The frozen test-set confusion matrix is:

```text
                 Predicted
              G1   G2   G3   G4

Actual G1    129   34   42   32
Actual G2     24  139   65   18
Actual G3     39   19  162   60
Actual G4     10    6   16  211
```

![Foot DFU Baseline — ResNet-50 Confusion Matrix](file:///d:/FusionMedAI/research/foot/Volume_04_Baseline_Framework/images/confusion_matrix.png)

The most prominent directional confusion is:
- **Grade 2 → Grade 3**: 65 samples
- **Grade 3 → Grade 2**: 19 samples

The Grade 2 / Grade 3 boundary will therefore receive particular attention during subsequent model benchmarking and error analysis.

---

## Error Analysis

The baseline produced **365 errors / 1,006 test samples**, corresponding to an error rate of approximately **36.28%**.

The complete prediction-level error analysis is archived in:
`experiments/foot/baseline_resnet50_unweighted/error_analysis.csv`

The analysis is intended to characterize failure modes rather than remove difficult samples or alter the frozen test population.

Potential relationships with the image-quality characteristics identified during Phase 10.3 are investigated separately from the baseline training objective.

---

## Reproducibility

The experiment uses deterministic dataset construction and seed-controlled training infrastructure.

The following are recorded with the experiment:
- Dataset split
- Source-group assignment
- Random seed (`42`)
- Model architecture (`ResNet-50`)
- Model initialization
- Training configuration
- Optimizer configuration
- Scheduler configuration
- Checkpoint state
- Training history
- Evaluation metrics

The baseline experiment is therefore intended to serve as a reproducible reference experiment rather than a final optimized Foot Ulcer model.
