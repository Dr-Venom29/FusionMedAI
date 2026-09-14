import torch
import torch.nn as nn
from typing import Optional
from src.foot.models.base_classifier import FootBaseClassifier
from src.foot.models.resnet50 import FootResNet50
from src.foot.models.efficientnet import FootEfficientNet

def create_model(
    name: str = "resnet50",
    num_classes: int = 4,
    pretrained: bool = True,
    dropout_rate: float = 0.2,
    device: Optional[str] = None
) -> FootBaseClassifier:
    """
    Factory function to instantiate Foot DFU classifiers.
    
    Args:
        name: Architecture name ('resnet50', 'efficientnet_b0', 'efficientnet')
        num_classes: Number of target output classes (default: 4)
        pretrained: Whether to load ImageNet pre-trained weights
        dropout_rate: Dropout rate for classifier head
        device: Target device ('cpu', 'cuda')
        
    Returns:
        Instance of FootBaseClassifier on specified device.
    """
    name_clean = name.lower().replace("-", "_")
    
    if name_clean in ("resnet50", "resnet_50", "resnet"):
        model = FootResNet50(num_classes=num_classes, pretrained=pretrained, dropout_rate=dropout_rate)
    elif name_clean in ("efficientnet_b0", "efficientnet", "effnet"):
        model = FootEfficientNet(num_classes=num_classes, pretrained=pretrained, dropout_rate=dropout_rate)
    else:
        raise ValueError(f"Unsupported model architecture '{name}'. Options: 'resnet50', 'efficientnet_b0'")
        
    if device is not None:
        model = model.to(device)
        
    return model

def build_foot_baseline_model(
    num_classes: int = 4,
    pretrained: bool = True,
    dropout_rate: float = 0.2,
    device: Optional[str] = None
) -> FootBaseClassifier:
    """Builds default ResNet-50 baseline classifier for Phase 10.4."""
    return create_model(
        name="resnet50",
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        device=device
    )
