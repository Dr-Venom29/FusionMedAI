# Chapter 8: Limitations and Future Work

The implemented data pipeline satisfies the functional requirements of the current development phase and has been verified through unit and end-to-end integration testing. However, several opportunities remain for improving robustness, scalability, and training efficiency.

## 1. Temporary ImageNet Normalization
- **Limitation**: The pipeline currently uses ImageNet mean ($\mu=[0.485, 0.456, 0.406]$) and standard deviation ($\sigma=[0.229, 0.224, 0.225]$) constants. Dataset-specific normalization statistics have not yet been computed and experimentally compared against ImageNet normalization.
- **Impact**: While acceptable for transfer learning with ImageNet-pretrained weights, these values do not match the true channel statistics of retinal fundus photographs, which contain a dominant red channel and much darker background pixels.
- **Future Work**: We plan to compute custom channel statistics over the APTOS 2019 training split and update these constants prior to final model training, evaluating their impact as an experimental variable.

## 2. Lack of Class-Weighted Sampling
- **Limitation**: The dataloader currently uses standard random sampling.
- **Impact**: Because the dataset is highly imbalanced (Class 0 is ~49%, while Class 3 is ~5%), standard random sampling will cause mini-batches to contain very few minority class examples. The model may struggle to learn features for severe disease stages.
- **Future Work**: `WeightedRandomSampler` addresses class imbalance during sampling, whereas Focal Loss addresses class imbalance during optimization. Future work includes evaluating sampling-based approaches (e.g., `WeightedRandomSampler`) and loss-based approaches (e.g., Focal Loss) independently to determine which provides better performance on minority disease stages.

## 3. Standard Augmentations Only
- **Limitation**: The pipeline relies on standard torchvision transformations (resize, rotate, flip, color jitter).
- **Impact**: Advanced augmentations are not currently implemented.
- **Future Work**: Advanced augmentation methods such as MixUp, CutMix, RandAugment, and Random Erasing can improve generalization by generating more diverse training examples and reducing overfitting. These techniques will be evaluated only after establishing a reliable baseline with conventional augmentations.

## 4. Sequential Loading in Verification Runs
- **Limitation**: The end-to-end integration test ran with `num_workers=0` (sequential execution on the main process).
- **Impact**: Loading 3,662 high-resolution images sequentially took ~578 seconds (elapsed time across all splits). While appropriate for debugging and Windows portability, this is too slow for actual model training.
- **Future Work**: The baseline framework has been completed. Systematic benchmarking across different worker counts will be performed during full-scale training experiments.

## 5. No Hardware-Accelerated Data Loading
- **Limitation**: The pipeline is CPU-bound, relying on Python's PIL and torchvision on the host.
- **Impact**: The CPU can become a bottleneck during training on high-end GPUs.
- **Future Work**: For larger datasets or faster training, we can investigate integrating hardware-accelerated libraries like NVIDIA DALI, Kornia, or torchvision v2 GPU transforms.

## 6. Corrupted Image Recovery
- **Limitation**: The current pipeline assumes that existing image files can be decoded successfully. Corrupted image files would raise an exception during loading.
- **Impact**: A single corrupted file could crash a long-running training job mid-epoch.
- **Future Work**: Introduce optional corrupted-image detection and logging to isolate problematic samples without interrupting large-scale training.

## 7. No Dataset Caching
- **Limitation**: Images are decoded from disk on every epoch.
- **Impact**: Repeated disk I/O increases training time.
- **Future Work**: Evaluate RAM caching, SSD caching, or WebDataset/Tensor storage formats for larger datasets.

## 8. Baseline Preprocessing Only
- **Limitation**: Advanced retinal preprocessing (CLAHE, Ben Graham preprocessing, circular cropping) remains outside the baseline pipeline.
- **Impact**: While basic resizing and normalization form a solid baseline, advanced domain-specific enhancements have not yet been evaluated.
- **Future Work**: Advanced retinal preprocessing will be evaluated experimentally after establishing EfficientNet-B0 baseline performance.

## 9. Single Baseline Architecture
- **Limitation**: The current implementation validates only EfficientNet-B0.
- **Impact**: We cannot compare model size, throughput, or accuracy across alternative architectures.
- **Future Work**: Future work includes benchmarking and evaluating alternative backbones, including EfficientNetV2, ConvNeXt, DenseNet, and Vision Transformers.

---

## Conclusion
Overall, the current pipeline provides a reproducible and modular foundation for model development. Future improvements will primarily focus on enhancing training efficiency, addressing class imbalance, and optimizing preprocessing for retinal imagery while preserving the modular architecture established in this phase.
