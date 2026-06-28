import sys
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.base_classifier import BaseClassifier

class EfficientNetB0(BaseClassifier):
    """
    EfficientNet-B0 classifier subclassing BaseClassifier.
    Loads ImageNet pretrained weights from torchvision and adapts the classification head.
    """
    
    def __init__(self, num_classes: int = 5, pretrained: bool = True) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        
        # 1. Load baseline backbone with dynamic weights check for compatibility
        if self.pretrained:
            try:
                # Modern torchvision (0.13+) API
                weights = models.EfficientNet_B0_Weights.DEFAULT
                self.backbone = models.efficientnet_b0(weights=weights)
            except (AttributeError, TypeError):
                # Fallback to legacy pretrained flag
                self.backbone = models.efficientnet_b0(pretrained=True)
        else:
            self.backbone = models.efficientnet_b0(weights=None)
            
        # 2. Extract input features count of the classifier
        in_features = self.backbone.classifier[1].in_features
        
        # 3. Replace the final linear head
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs the full forward pass to obtain logits.
        """
        return self.backbone(x)
        
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts high-dimensional visual representation features before the pooling/classifier.
        Useful for Grad-CAM and multimodal joint cross-attention.
        """
        return self.backbone.features(x)
