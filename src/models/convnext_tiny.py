import sys
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.base_classifier import BaseClassifier

class ConvNeXtTiny(BaseClassifier):
    """
    ConvNeXt Tiny classifier subclassing BaseClassifier.
    Loads ImageNet pretrained weights from torchvision and adapts the classification head.
    """
    
    def __init__(self, num_classes: int = 5, pretrained: bool = True) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        
        # 1. Load backbone with dynamic weights check for compatibility
        if self.pretrained:
            try:
                # Modern torchvision (0.13+) API
                weights = models.ConvNeXt_Tiny_Weights.DEFAULT
                self.backbone = models.convnext_tiny(weights=weights)
            except (AttributeError, TypeError):
                # Fallback to legacy pretrained flag
                self.backbone = models.convnext_tiny(pretrained=True)
        else:
            self.backbone = models.convnext_tiny(weights=None)
            
        # 2. Replace the classification head
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Linear(in_features, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs the full forward pass to obtain logits.
        """
        return self.backbone(x)
        
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts high-dimensional visual representation features before classification.
        Returns shape: [batch_size, 768, H_feat, W_feat]
        """
        return self.backbone.features(x)
