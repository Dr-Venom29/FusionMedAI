# Chapter 4: Image Transforms

## Why Augmentation is Needed
Retinal fundus images are complex medical representations containing subtle structures like blood vessels, hemorrhages, and microaneurysms. Clinical datasets are generally small, which can lead to rapid overfitting.
Data Augmentation artificially expands the dataset size by applying random geometric and color perturbations to the training set. This forces the neural network to learn robust, invariant features—encouraging the network to become robust to variations introduced during image acquisition, including small changes in camera orientation, illumination, and color response—rather than memorizing trivial background noise.

## Transform Pipelines
Three separate transform pipelines are exposed through API functions:

```
  Training Transforms                       Validation & Test Transforms
 ┌─────────────────────┐                   ┌─────────────────────┐
 │    Raw PIL Image    │                   │    Raw PIL Image    │
 └──────────┬──────────┘                   └──────────┬──────────┘
            │                                         │
            ▼                                         ▼
 ┌─────────────────────┐                   ┌─────────────────────┐
 │  Resize (Bilinear)  │                   │  Resize (Bilinear)  │
 └──────────┬──────────┘                   └──────────┬──────────┘
            │                                         │
            ▼                                         │
 ┌─────────────────────┐                              │
 │   Random Rotation   │                              │
 └──────────┬──────────┘                              │
            │                                         │
            ▼                                         │
 ┌─────────────────────┐                              │
 │   Random Hor.Flip   │                              │
 └──────────┬──────────┘                              │
            │                                         │
            ▼                                         │
 ┌─────────────────────┐                              │
 │    Color Jitter     │                              │
 └──────────┬──────────┘                              │
            │                                         │
            ▼                                         ▼
 ┌─────────────────────┐                   ┌─────────────────────┐
 │      ToTensor       │                   │      ToTensor       │
 └──────────┬──────────┘                   └──────────┬──────────┘
            │                                         │
            ▼                                         ▼
 ┌─────────────────────┐                   ┌─────────────────────┐
 │      Normalize      │                   │      Normalize      │
 └─────────────────────┘                   └─────────────────────┘
```

### 1. Training Pipeline
Designed to introduce diverse samples for training:
- **Resize**: Resizes images to a standard dimension.
- **Random Rotation**: Randomly rotates the image by $\pm 15^\circ$ to simulate slight variations in head positioning during fundus photography.
- **Random Horizontal Flip**: Flips the image horizontally ($p=0.5$). Horizontal flipping is commonly used in retinal image analysis because it preserves local pathological structures while increasing viewpoint diversity. The clinical diagnosis depends primarily on lesion appearance rather than global left-right orientation.
- **Color Jitter**: Slightly perturbs brightness, contrast, saturation, and hue to simulate variance in lighting, flash intensity, and sensor response.
- **ToTensor**: Converts the PIL Image to a float32 PyTorch tensor and scales pixel values to $[0.0, 1.0]$.
- **Normalize**: Z-score normalizes the image channel-wise.

Vertical flipping was intentionally omitted because inverted fundus photographs are not representative of normal clinical image acquisition and may introduce unrealistic anatomical orientations.

### 2. Validation & Test Pipelines
Deterministic pipelines for evaluation. No augmentations are applied. This guarantees that model evaluation is consistent and reproducible. The test pipeline is configured to call `get_val_transforms()` directly, ensuring they never diverge.
Applying stochastic augmentations during evaluation would introduce variability unrelated to model performance, making comparisons between experiments unreliable.

## Preprocessing Rationale

### Resize
All images are resized to $224 \times 224$ pixels using `InterpolationMode.BILINEAR` interpolation:
- Standardized shapes are necessary to build batched tensors.
- Bilinear interpolation provides a good balance between speed and quality, preserving important details of diabetic lesions. It is also the default interpolation strategy recommended for natural-image pretrained CNN backbones.
- **Implementation Note**: Interpolation mode is specified explicitly rather than relying on library defaults, ensuring consistent behavior across different torchvision versions.

### Normalization and ImageNet Statistics
The normalization transform subtracts a mean vector and divides by a standard deviation vector:
$$x_{norm} = \frac{x - \mu}{\sigma}$$
- **ImageNet Statistics**: Currently, the pipeline utilizes ImageNet statistics:
  - $\mu = [0.485, 0.456, 0.406]$
  - $\sigma = [0.229, 0.224, 0.225]$
- **Rationale**: Since the model will initialize with ImageNet-pretrained weights (e.g., EfficientNet-B0), using the pretrained statistics matches the distribution of the features the network has already learned.
- **Wording Refinement**: These statistics are currently adopted to maintain compatibility with ImageNet-pretrained weights. Dataset-specific normalization statistics may be computed and substituted in future experiments.

The preprocessing configuration presented here represents the baseline transform pipeline. Subsequent EDA findings may recommend adjustments to resizing strategy, normalization statistics, augmentation strength, or preprocessing order based on observed dataset characteristics.

## Implementation Characteristics
The transform module provides:
- **Separate pipelines for training and evaluation**: Ensuring safe benchmarks.
- **Modular API-based transform construction**: Exposes Compose callables.
- **Deterministic validation and testing**: Guarantees zero testing jitter.
- **Stochastic augmentation during training**: Regularizes the optimization path.
- **Fixed output resolution**: Outputs $224 \times 224$ pixel tensors.
- **Standardized tensor normalization**: Aligns input distribution with pretrained baselines.

---

## References
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. *Advances in Neural Information Processing Systems*, 25, 1097-1105.
- Cubuk, E. D., Zoph, B., Mane, D., Vasudevan, V., & Le, Q. V. (2018). Autoaugment: Learning augmentation policies from data. *arXiv preprint arXiv:1805.09501*.
