# Chapter 02 — Explainability Method & Target Layer Selection

## 2.1 Mathematical Formulation of Grad-CAM

Gradient-weighted Class Activation Mapping (Grad-CAM) computes a coarse localization map highlighting important regions in an input image $X$ for predicting a specific class score $y^c$.

For target class $c$, let $A^k$ represent the $k$-th feature map of a chosen convolutional layer, and let $y^c$ be the unnormalized class score (logit) for class $c$.

### Step 1: Neuron Importance Weights $\alpha_k^c$
The weight $\alpha_k^c$ represents the importance of feature map $A^k$ for class $c$, calculated via global average pooling of the gradients:

$$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^{H} \sum_{j=1}^{W} \frac{\partial y^c}{\partial A_{i,j}^k}$$

where $Z = H \times W$ is the spatial area of the feature map.

### Step 2: Weighted Combination & ReLU
The activation map $L_{\text{Grad-CAM}}^c$ is computed as a weighted combination of forward activation maps followed by a Rectified Linear Unit (ReLU) to filter out negative attributions:

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$

### Step 3: Spatial Normalization & Resizing
The resulting 2D map is min-max normalized to $[0, 1]$ and bilinearly interpolated to the original input resolution $(224 \times 224)$:

$$S^c(x, y) = \text{BilinearResize}\left( \frac{L^c - \min(L^c)}{\max(L^c) - \min(L^c) + \epsilon}, (224, 224) \right)$$

---

## 2.2 Target Layer Verification for EfficientNet-B3

To avoid arbitrary target layer selection, we inspect the torchvision `EfficientNet-B3` model hierarchy:

- `backbone.features[0]`: Initial Conv2d stem ($3 \to 40$, stride 2)
- `backbone.features[1..7]`: MBConv residual bottleneck blocks
- `backbone.features[8]`: Final `Conv2dNormActivation` block ($384 \to 1536$, $1 \times 1$ conv + BN + SiLU)
- `backbone.avgpool`: `AdaptiveAvgPool2d(output_size=1)`
- `backbone.classifier`: Sequential(`Dropout(p=0.2)`, `Linear(1536, 4)`)

### Verification Result
Hooking `backbone.features[8]` yields a spatial feature tensor of shape `[B, 1536, 7, 7]`. This represents the final spatial representation containing maximal high-level semantic features before spatial collapsing by average pooling.

Therefore, `target_layer = "backbone.features.8"` is selected and frozen in the experiment configuration.
