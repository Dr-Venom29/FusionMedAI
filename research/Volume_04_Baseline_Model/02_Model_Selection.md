# Chapter 2: Model Selection Rationale

Selecting an appropriate baseline architecture is a critical step in developing a reliable medical image classification system. The baseline model should provide strong predictive performance while remaining computationally efficient, reproducible, and suitable for systematic comparison with more advanced architectures. Based on these criteria, **EfficientNet-B0** was selected as the baseline convolutional neural network for Diabetic Retinopathy severity classification.

## Rationale

### 1. Computational Efficiency

EfficientNet-B0 employs the **Compound Scaling** strategy, which jointly scales network depth, width, and input resolution using a balanced scaling coefficient. This design achieves an excellent trade-off between classification performance and computational cost while maintaining a compact model size of approximately **4.01 million trainable parameters (~15 MB)** after replacing the original ImageNet classifier with a five-class Diabetic Retinopathy classification head. Its lightweight design makes it suitable for rapid experimentation and future deployment on resource-constrained clinical systems.

### 2. Transfer Learning

The model is initialized using **ImageNet pretrained weights**, allowing the network to leverage generic low-level visual representations such as edges, textures, and shapes. Fine-tuning these pretrained features significantly accelerates convergence and typically improves performance compared with training from random initialization, particularly when working with relatively small medical imaging datasets such as APTOS 2019.

### 3. Standardized PyTorch Implementation

The baseline implementation uses the official **`torchvision.models.efficientnet_b0`** architecture rather than third-party implementations. This reduces external dependencies, ensures long-term compatibility with the PyTorch ecosystem, and provides a stable foundation for reproducible experimentation and future maintenance.

### 4. Reference Baseline for Comparative Studies

The primary purpose of EfficientNet-B0 is to establish a **reference baseline** for subsequent experimental studies. Future phases of the project will compare this baseline against more advanced backbone architectures, including **EfficientNet-B3, ConvNeXt, Swin Transformer, and Vision Transformer (ViT)**, using the same training framework and evaluation protocol. This enables fair comparison of accuracy, computational complexity, inference latency, and clinically relevant metrics such as Quadratic Weighted Kappa (QWK) and Macro F1-score.

### 5. Extensible Framework Design

Although EfficientNet-B0 is the initial backbone, the training framework has been intentionally designed to be model-agnostic through the use of a common `BaseClassifier` interface and a dynamic model factory. Consequently, new architectures can be integrated with minimal changes to the surrounding training, evaluation, inference, and experiment management pipelines.
