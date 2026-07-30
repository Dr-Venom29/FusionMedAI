# 02 GradCAM Implementation

FusionMedAI implements its Class Activation Mapping (CAM) logic entirely from scratch. This custom implementation provides precise control over hook lifecycle management, normalization, and tensor extraction, avoiding the opacity of third-party libraries.

## BaseCAM Abstract Class

`src/xai/cam.py` defines the `BaseCAM` class, which handles:
1.  **Lifecycle Management**: Dynamically registers forward and full-backward PyTorch hooks onto the target convolutional layer during initialization, and cleanly clears them via `remove_hooks()` upon destruction to prevent memory leaks.
2.  **Normalization**: Normalizes the raw activation arrays linearly to a `[0, 1]` range.
3.  **Visualization Overlay**: Fuses the generated heatmap with the original image using a predefined colormap and alpha transparency.

## Grad-CAM and Grad-CAM++

*   **Grad-CAM (`src/xai/gradcam.py`)**: Computes the channel-wise gradients of the target class score with respect to the feature map activations, pools them globally, and computes a linear combination of the activations.
*   **Grad-CAM++ (`src/xai/gradcampp.py`)**: An extension of standard Grad-CAM that applies higher-order derivatives (alpha coefficients) to weight the gradients. This results in significantly better localization when an image contains multiple instances of a class or scattered features.

## Enhancements for Readability

Raw CAM outputs can often appear pixelated or blocky when upsampled from a small feature map (e.g., $7 \times 7$) to a high-resolution image ($512 \times 512$). To combat this, FusionMedAI applies a `cv2.GaussianBlur` smoothing pass to the raw CAM array *before* resizing, yielding a significantly cleaner and more interpretable topographical heatmap.

## Validation

The Grad-CAM implementation was successfully validated over the complete APTOS 2019 test set and produced smooth heatmaps after Gaussian filtering for all selected representative cases.
