import torch
import torch.nn as nn
from pathlib import Path
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

FINAL_FOOT_MODEL_CHECKPOINT = Path(__file__).resolve().parents[3] / "experiments" / "foot" / "architecture_benchmark" / "efficientnet_b3" / "checkpoints" / "best_model.pt"

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

def build_foot_final_model(
    num_classes: int = 4,
    pretrained: bool = True,
    dropout_rate: float = 0.2,
    device: Optional[str] = None
) -> FootBaseClassifier:
    """Builds the selected Phase 10.5 primary Foot Ulcer classification model (EfficientNet-B3)."""
    return create_model(
        model_name="efficientnet_b3",
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        device=device
    )

def load_foot_final_model(
    checkpoint_path: Optional[Path] = None,
    device: str = "cpu"
) -> FootBaseClassifier:
    """
    Loads the trained and frozen Phase 10.5 Foot Ulcer primary model (EfficientNet-B3)
    from its best checkpoint and returns it in evaluation mode.
    """
    path = Path(checkpoint_path) if checkpoint_path is not None else FINAL_FOOT_MODEL_CHECKPOINT
    if not path.exists():
        raise FileNotFoundError(f"Final Foot model checkpoint not found at: '{path}'")
        
    model = build_foot_final_model(num_classes=4, pretrained=False, dropout_rate=0.2, device=device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
