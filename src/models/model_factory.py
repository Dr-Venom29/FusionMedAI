import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.base_classifier import BaseClassifier
from src.models.efficientnet_b0 import EfficientNetB0
from src.models.efficientnet_b3 import EfficientNetB3
from src.models.convnext_tiny import ConvNeXtTiny
from src.models.swin_tiny import SwinTiny
from src.models.vit_b16 import ViTB16

def load_model(name: str, num_classes: int = 5, pretrained: bool = True) -> BaseClassifier:
    """
    Factory function to dynamically instantiate registered model architectures.
    
    Args:
        name: Name of the model architecture (e.g. 'efficientnet_b0').
        num_classes: Number of classes for classification head.
        pretrained: If True, loads pre-trained ImageNet weights.
        
    Returns:
        BaseClassifier: Instantiated model wrapper.
        
    Raises:
        ValueError: If model name is not supported.
    """
    model_name_lower = name.lower().replace("-", "_")
    
    if model_name_lower == "efficientnet_b0":
        return EfficientNetB0(num_classes=num_classes, pretrained=pretrained)
    elif model_name_lower == "efficientnet_b3":
        return EfficientNetB3(num_classes=num_classes, pretrained=pretrained)
    elif model_name_lower == "convnext_tiny":
        return ConvNeXtTiny(num_classes=num_classes, pretrained=pretrained)
    elif model_name_lower == "swin_tiny" or model_name_lower == "swin_t":
        return SwinTiny(num_classes=num_classes, pretrained=pretrained)
    elif model_name_lower == "vit_b16" or model_name_lower == "vit_b_16":
        return ViTB16(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(
            f"Unsupported model architecture: '{name}'. "
            f"Currently registered architectures: ['efficientnet_b0', 'efficientnet_b3', 'convnext_tiny', 'swin_tiny', 'vit_b16']"
        )
