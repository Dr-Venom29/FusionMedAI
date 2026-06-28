# Chapter 7: Learning Rate Scheduling

The baseline EfficientNet-B0 model employs the **Cosine Annealing Learning Rate Scheduler** (`torch.optim.lr_scheduler.CosineAnnealingLR`) to gradually reduce the learning rate throughout training. Learning rate scheduling plays a critical role in transfer learning by allowing rapid adaptation during the initial training stages while enabling stable fine-tuning as optimization converges.

## Baseline Configuration

| Hyperparameter                           |        Value       | Purpose                                                          |
| :--------------------------------------- | :----------------: | :--------------------------------------------------------------- |
| **Scheduler**                            |  CosineAnnealingLR | Smooth learning rate decay                                       |
| **Maximum Epochs ($T_{max}$)**           |         20         | One complete cosine decay cycle                                  |
| **Minimum Learning Rate ($\eta_{min}$)** | $1 \times 10^{-6}$ | Maintains small parameter updates during late-stage optimization |

## Selection Rationale

### 1. Smooth Learning Rate Decay

Cosine Annealing decreases the learning rate gradually rather than introducing abrupt changes. This smooth transition promotes stable optimization and reduces oscillations near convergence.

### 2. Effective Transfer Learning

During the early training epochs, a relatively larger learning rate allows the pretrained EfficientNet-B0 backbone to adapt efficiently to retinal fundus images. As training progresses, the learning rate decreases continuously, enabling increasingly fine parameter updates.

### 3. Fine-Tuning in Later Epochs

The minimum learning rate is fixed at **$1 \times 10^{-6}$** instead of reaching zero. This allows optimization to continue making small but meaningful parameter updates during the final training epochs rather than completely stopping learning.

### 4. Single-Cycle Baseline Schedule

For the baseline implementation, the scheduler cycle length ($T_{max}$) is set equal to the total number of training epochs. This produces one complete cosine decay curve during the entire training process and provides a simple, reproducible scheduling strategy suitable for baseline benchmarking.

## Alternative Scheduling Strategies

The learning rate scheduler is implemented through a modular scheduler factory, allowing alternative scheduling methods to be incorporated without modifying the training pipeline.

Future experimental studies will compare Cosine Annealing with:

* **ReduceLROnPlateau**, which adapts the learning rate based on validation performance.
* **OneCycleLR**, designed for rapid convergence through cyclic learning rate policies.
* **StepLR**, which decreases the learning rate at predefined intervals.

These schedulers will be evaluated under identical training conditions during Step 5 to determine their impact on convergence speed, Quadratic Weighted Kappa (QWK), and overall classification performance.

## Design Considerations

The baseline scheduling strategy emphasizes:

* Stable optimization during transfer learning.
* Smooth convergence without abrupt learning rate transitions.
* Reproducibility across repeated experiments.
* Fair comparison with future scheduling strategies.

By separating scheduler configuration from the training loop, the framework supports systematic experimentation while maintaining a clean and modular software architecture.
