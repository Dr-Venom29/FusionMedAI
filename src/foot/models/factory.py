import torch
import torch.nn as nn
from typing import Optional, Dict, Type

from src.foot.models.base_classifier import FootBaseClassifier
from src.foot.models.resnet50 import FootResNet50
from src.foot.models.architectures import (
    FootEfficientNetB0,
    FootEfficientNetB3,
    FootConvNeXtTiny,
    FootSwinTiny,
    FootViTB16
)

MODEL_REGISTRY: Dict[str, Type[FootBaseClassifier]] = {
    "resnet50": FootResNet50,
    "efficientnet_b0": FootEfficientNetB0,
    "efficientnet_b3": FootEfficientNetB3,
    "convnext_tiny": FootConvNeXtTiny,
    "swin_tiny": FootSwinTiny,
    "vit_b16": FootViTB16
}

def create_model(
    model_name: str = "resnet50",
    num_classes: int = 4,
    pretrained: bool = True,
    dropout_rate: float = 0.2,
    device: Optional[str] = None
) -> FootBaseClassifier:
    """
    Central Factory Function for Phase 10.5 Architecture Benchmarking (10.5.3).
    
    Args:
        model_name: Name of model architecture in MODEL_REGISTRY
        num_classes: Number of target Wagner classes (default: 4)
        pretrained: Whether to load ImageNet pre-trained weights
        dropout_rate: Dropout rate for classification head
        device: Device to place model ('cpu' or 'cuda')
        
    Returns:
        Instance of FootBaseClassifier on target device.
    """
    name_clean = model_name.lower().replace("-", "_")
    
    if name_clean not in MODEL_REGISTRY:
        supported = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unsupported model architecture '{model_name}'. Options: {supported}")
        
    model_cls = MODEL_REGISTRY[name_clean]
    model = model_cls(num_classes=num_classes, pretrained=pretrained, dropout_rate=dropout_rate)
    
    if device is not None:
        target_device = torch.device(device) if isinstance(device, str) else device
        model = model.to(target_device)
        
    return model

def build_foot_baseline_model(
    num_classes: int = 4,
    pretrained: bool = True,
    dropout_rate: float = 0.2,
    device: Optional[str] = None
) -> FootBaseClassifier:
    """Builds default ResNet-50 baseline classifier for Phase 10.4."""
    return create_model(
        model_name="resnet50",
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        device=device
    )
