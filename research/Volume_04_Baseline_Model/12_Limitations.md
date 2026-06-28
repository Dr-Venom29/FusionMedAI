# Chapter 12: Limitations of the Baseline Model

Although the baseline EfficientNet-B0 framework establishes a reproducible and modular foundation for Diabetic Retinopathy severity classification, several limitations remain. These limitations are intentional, as the primary objective of Step 4 is to establish a reliable reference implementation rather than maximize predictive performance.

The identified limitations define the scope of future experimental investigations.

## 1. Input Resolution Constraints

All retinal fundus images are resized to **224 × 224** pixels to match the default EfficientNet-B0 input resolution.

While this configuration provides efficient training and inference, resizing high-resolution retinal images may reduce the visibility of subtle pathological structures such as microaneurysms, hemorrhages, and fine vascular abnormalities.

Future experiments will evaluate larger input resolutions (e.g., 300 × 300 and 384 × 384) to determine the trade-off between computational cost and diagnostic performance.

---

## 2. Baseline Loss Function

The baseline implementation employs standard Cross Entropy Loss.

Although appropriate for establishing a reproducible reference baseline, Cross Entropy treats every misclassification equally and does not explicitly model:

* Ordinal disease severity
* Class imbalance
* Hard example emphasis

Future experiments will investigate Weighted Cross Entropy, Focal Loss, and ordinal-aware objective functions.

---

## 3. Baseline Preprocessing Strategy

The baseline pipeline intentionally applies only standard preprocessing and normalization.

Advanced retinal preprocessing techniques—including circular fundus cropping, black-border removal, CLAHE, Ben Graham preprocessing, and illumination normalization—have been intentionally excluded from the baseline to ensure that architectural performance is evaluated under a simple and reproducible preprocessing pipeline.

These preprocessing strategies will be evaluated independently during future experimental studies.

---

## 4. Single Baseline Architecture

The current implementation validates only the EfficientNet-B0 backbone.

Although the training framework supports multiple architectures through the common `BaseClassifier` interface and model factory, comparative evaluation of EfficientNet-B3, ConvNeXt, Swin Transformer, and Vision Transformer has not yet been performed.

These architecture comparisons are planned for subsequent experimental phases.

---

## 5. Image-Only Learning

The current Retina Module operates exclusively on retinal fundus images.

Additional clinical information—including demographic variables, laboratory measurements, electronic health records, and ophthalmology reports—is not incorporated into the baseline model.

Multimodal integration will be introduced later through the ACARA-U Fusion framework.

---

## Summary

```mermaid
flowchart TD
    subgraph Current [Step 4 Baseline]
        direction LR
        A1[EfficientNet-B0] --> A2[Cross Entropy] --> A3[224x224 Input] --> A4[Image Only]
    end
    subgraph Future [Step 5 & Beyond]
        direction LR
        B1[ConvNeXt / ViT] --> B2[Focal / Ordinal Loss] --> B3[Advanced Preprocessing] --> B4[Multimodal Fusion]
    end
```

The baseline EfficientNet-B0 implementation intentionally prioritizes simplicity, reproducibility, and modularity over maximum predictive performance. Establishing this stable reference framework enables future improvements to be evaluated through controlled experiments rather than simultaneous changes to multiple system components.

Consequently, subsequent phases of the FusionMedAI project will focus on hyperparameter optimization, advanced preprocessing, architecture comparison, explainability, calibration, uncertainty estimation, and multimodal fusion while maintaining the baseline framework established in this phase.
