# Chapter 3: Selected Architectures

## 3.1 EfficientNet-B0
- **Architecture**: CNN (Mobile Inverted Bottlenecks, Squeeze-and-Excitation)
- **Advantages**: Highly optimized for efficiency and speed; compound scaling.
- **Parameter Count**: ~4.01M

## 3.2 EfficientNet-B3
- **Architecture**: CNN (Scaled up from B0)
- **Advantages**: Better feature extraction capacity than B0 while retaining reasonable efficiency.
- **Parameter Count**: ~10.70M

## 3.3 ConvNeXt-Tiny
- **Architecture**: CNN (Modernized ResNet)
- **Advantages**: Integrates design lessons from Vision Transformers into a pure CNN.
- **Parameter Count**: ~27.82M

## 3.4 Swin-Tiny
- **Architecture**: Hierarchical Vision Transformer
- **Advantages**: Computes self-attention within shifted local windows, providing spatial hierarchies.
- **Parameter Count**: ~27.52M

## 3.5 ViT-B/16 (Vision Transformer)
- **Architecture**: Vision Transformer
- **Advantages**: Processes images as a sequence of patches with global self-attention.
- **Parameter Count**: ~85.80M

## 3.6 Selected Architectures Overview

| Model | Type | Params | Primary Motivation |
| :--- | :--- | :--- | :--- |
| EfficientNet-B0 | CNN | 4.01M | Lightweight baseline |
| EfficientNet-B3 | CNN | 10.70M | Accuracy |
| ConvNeXt-Tiny | CNN | 27.82M | Modern CNN |
| Swin-Tiny | Transformer | 27.52M | Hierarchical attention |
| ViT-B/16 | Transformer | 85.80M | Global attention |
