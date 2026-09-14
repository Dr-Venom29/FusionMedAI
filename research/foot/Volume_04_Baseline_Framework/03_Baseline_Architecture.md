# 03 Baseline Architecture — ResNet-50

## Architecture Specification

- **Backbone**: `ResNet-50` (torchvision pre-trained ImageNet weights)
- **Feature Layer**: Standard 2048-dimensional global average pooling output
- **Classification Head**: `nn.Sequential(nn.Dropout(p=0.2), nn.Linear(2048, 4))`
- **Output Interface**:
  ```python
  {
      "logits": Tensor [B, 4],
      "probs": Softmax Probabilities [B, 4],
      "predicted_class": Argmax Class Indices [B]
  }
  ```
- **Parameter Count**: ~23.5 Million parameters
- **Implementation**: Defined in `src/foot/models/resnet50.py` and instantiated via `src/foot/models/factory.py`.
