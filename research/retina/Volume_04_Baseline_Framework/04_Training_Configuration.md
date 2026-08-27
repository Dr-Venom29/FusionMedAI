# Chapter 4: Training Configuration

The baseline training configuration is designed to provide stable transfer learning while establishing a reproducible reference point for future experiments. This chapter details the objective function, optimization algorithm, and learning rate scheduling strategy.

## 4.1 Loss Function

The baseline model is trained using the standard **Cross Entropy Loss** (`torch.nn.CrossEntropyLoss`). Cross Entropy is the most widely adopted objective function for multi-class image classification. 

Although Diabetic Retinopathy severity follows an ordinal progression (No DR → Mild → Moderate → Severe → Proliferative DR), the baseline implementation intentionally formulates the task as a conventional five-class classification problem. This provides a reproducible reference point before introducing more specialized loss functions.

### Selection Rationale
* **Standard Research Baseline**: Widely reported in the diabetic retinopathy literature, establishing a strong baseline.
* **Stable Optimization**: Provides stable gradients across all output classes, allowing the pretrained backbone to adapt effectively to retinal fundus images without requiring loss-specific hyperparameter tuning.
* **Compatibility with Transfer Learning**: Integrates naturally with the ImageNet fine-tuning strategy.

## 4.2 Optimizer

The model is optimized using the **AdamW** optimizer. AdamW combines the adaptive learning rate mechanism of Adam with decoupled weight decay regularization, providing stable optimization and improved generalization during transfer learning. For the baseline implementation, all network parameters are fine-tuned without freezing any layers.

| Hyperparameter       |        Value       | Purpose                                              |
| :------------------- | :----------------: | :--------------------------------------------------- |
| **Optimizer**        |        AdamW       | Adaptive optimization with decoupled weight decay    |
| **Learning Rate**    | $1 \times 10^{-4}$ | Stable fine-tuning of pretrained weights             |
| **Weight Decay**     | $1 \times 10^{-4}$ | Reduces overfitting through parameter regularization |
| **Trainable Layers** |         All        | Complete backbone adaptation                         |

### Selection Rationale
* **Stable Transfer Learning**: A relatively small learning rate ($1 \times 10^{-4}$) enables gradual adaptation of pretrained representations while minimizing catastrophic forgetting.
* **Decoupled Weight Decay**: Provides more effective regularization and generally improves model generalization compared to standard Adam.
* **Full Backbone Fine-Tuning**: Retinal fundus images differ substantially from natural images, making full-network adaptation more appropriate than freezing early layers.

## 4.3 Scheduler

The training pipeline employs the **Cosine Annealing Learning Rate Scheduler** (`torch.optim.lr_scheduler.CosineAnnealingLR`) to gradually reduce the learning rate throughout training.

| Hyperparameter                           |        Value       | Purpose                                                          |
| :--------------------------------------- | :----------------: | :--------------------------------------------------------------- |
| **Scheduler**                            |  CosineAnnealingLR | Smooth learning rate decay                                       |
| **Maximum Epochs ($T_{max}$)**           |         20         | One complete cosine decay cycle                                  |
| **Minimum Learning Rate ($\eta_{min}$)** | $1 \times 10^{-6}$ | Maintains small parameter updates during late-stage optimization |

### Selection Rationale
* **Smooth Learning Rate Decay**: Decreases the learning rate gradually rather than abruptly, promoting stable optimization and reducing oscillations near convergence.
* **Effective Transfer Learning**: A larger initial learning rate allows efficient adaptation, while continuous decay enables increasingly fine parameter updates.
* **Single-Cycle Baseline**: Setting the cycle length ($T_{max}$) equal to the total number of training epochs produces one complete cosine decay curve, providing a simple and reproducible baseline.

## 4.4 Future Alternatives

The training framework exposes configurable factories (`get_loss_fn()`, `get_optimizer()`, `get_scheduler()`), allowing alternative configurations to be incorporated without modifying the core training loop. Future experimental phases will benchmark these components:

* **Loss Functions**: **Weighted Cross Entropy** (to compensate for class imbalance), **Focal Loss** (to emphasize difficult minority-class examples), and **Ordinal Loss Functions** (to explicitly model the ordered disease progression).
* **Optimizers**: **Adam** (to evaluate the impact of standard weight decay) and **Stochastic Gradient Descent (SGD)** with momentum.
* **Schedulers**: **ReduceLROnPlateau** (validation-based adaptation), **OneCycleLR** (cyclic policies), and **StepLR** (interval-based decay).

These methods were intentionally deferred until after the baseline framework was fully established and evaluated.
