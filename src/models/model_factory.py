import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.base_classifier import BaseClassifier
from src.models.efficientnet_b0 import EfficientNetB0

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
    else:
        raise ValueError(
            f"Unsupported model architecture: '{name}'. "
            f"Currently registered architectures: ['efficientnet_b0']"
        )
