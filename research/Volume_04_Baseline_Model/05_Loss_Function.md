# Chapter 5: Loss Function Rationale

The baseline EfficientNet-B0 model is trained using the standard **Cross Entropy Loss** implemented through `torch.nn.CrossEntropyLoss`. Cross Entropy is the most widely adopted objective function for multi-class image classification and serves as an appropriate baseline for establishing the initial performance of the Retina Module.

Although Diabetic Retinopathy severity follows an ordinal progression (No DR → Mild → Moderate → Severe → Proliferative DR), the baseline implementation intentionally formulates the task as a conventional five-class classification problem. This provides a reproducible reference point before introducing more specialized loss functions in later experimental phases.

## Selection Rationale

### 1. Standard Research Baseline

Cross Entropy Loss is the de facto standard loss function for multi-class image classification and is widely reported in the diabetic retinopathy literature. Using this loss establishes a strong and reproducible baseline against which future improvements can be fairly evaluated.

### 2. Stable Optimization

During transfer learning, Cross Entropy provides stable gradients across all output classes, allowing the pretrained EfficientNet-B0 backbone to adapt effectively to retinal fundus images without requiring additional loss-specific hyperparameter tuning.

### 3. Compatibility with Transfer Learning

The EfficientNet-B0 backbone is initialized using ImageNet pretrained weights. Cross Entropy integrates naturally with this fine-tuning strategy and avoids introducing additional optimization complexity during the baseline phase.

### 4. Modular Loss Framework

The training framework exposes a configurable `get_loss_fn()` factory, allowing alternative objective functions to be incorporated without modifying the training loop. This modular design supports future experimentation while keeping the baseline implementation simple and reproducible.

## Future Extensions

The APTOS 2019 dataset exhibits class imbalance, particularly for the Severe and Proliferative Diabetic Retinopathy categories. Furthermore, disease severity represents an ordinal classification problem in which neighboring classes are clinically more similar than distant classes.

To address these characteristics, future experimental phases will evaluate:

* **Weighted Cross Entropy** to compensate for class imbalance.
* **Focal Loss** to emphasize difficult and minority-class examples.
* **Ordinal Loss Functions** that explicitly model the ordered relationship between disease severity stages.

These methods are intentionally deferred until after the baseline model has been fully trained and evaluated, ensuring that all subsequent improvements can be compared against a consistent reference baseline.
