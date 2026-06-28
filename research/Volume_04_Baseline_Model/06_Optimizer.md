# Chapter 6: Optimizer Configuration

The baseline EfficientNet-B0 model is optimized using the **AdamW** optimizer. AdamW combines the adaptive learning rate mechanism of Adam with decoupled weight decay regularization, providing stable optimization and improved generalization during transfer learning.

For the baseline implementation, all network parameters are fine-tuned without freezing any layers, allowing the pretrained ImageNet backbone to fully adapt to retinal fundus images.

## Baseline Configuration

| Hyperparameter       |        Value       | Purpose                                              |
| :------------------- | :----------------: | :--------------------------------------------------- |
| **Optimizer**        |        AdamW       | Adaptive optimization with decoupled weight decay    |
| **Learning Rate**    | $1 \times 10^{-4}$ | Stable fine-tuning of pretrained weights             |
| **Weight Decay**     | $1 \times 10^{-4}$ | Reduces overfitting through parameter regularization |
| **Trainable Layers** |         All        | Complete backbone adaptation                         |

## Selection Rationale

### 1. Stable Transfer Learning

The EfficientNet-B0 backbone is initialized using ImageNet pretrained weights. A relatively small learning rate of **$1 \times 10^{-4}$** enables gradual adaptation of these pretrained representations while minimizing the risk of catastrophic forgetting.

### 2. Decoupled Weight Decay

Unlike the original Adam optimizer, AdamW decouples weight decay from the gradient update process. This provides more effective regularization and generally improves model generalization, particularly during fine-tuning of deep convolutional networks.

### 3. Full Backbone Fine-Tuning

Rather than freezing early convolutional layers, all parameters are updated during training. Retinal fundus images differ substantially from natural images used during ImageNet pretraining, making full-network adaptation more appropriate for the baseline model.

### 4. Baseline Consistency

AdamW is widely adopted in modern computer vision research and provides a strong optimization baseline. Using a standard optimizer ensures that future improvements can be attributed to architectural or methodological changes rather than optimizer selection.

## Alternative Optimizers

Although AdamW is selected for the baseline implementation, the optimizer framework is intentionally modular through the `get_optimizer()` factory.

Future experimental phases will benchmark:

* **Adam** — to evaluate the impact of decoupled weight decay.
* **Stochastic Gradient Descent (SGD)** with momentum — to compare adaptive and non-adaptive optimization strategies.

These comparisons will be conducted under identical training conditions during Step 5 to determine the optimizer that provides the best balance between convergence speed and final classification performance.

## Design Considerations

The baseline optimizer configuration prioritizes:

* Stable convergence during transfer learning.
* Explicit regularization through decoupled weight decay.
* Complete adaptation of the pretrained backbone.
* Reproducibility and fair comparison with future optimization strategies.

The modular optimizer factory enables additional optimization algorithms to be incorporated without modifying the surrounding training framework, ensuring extensibility for subsequent experimental studies.
