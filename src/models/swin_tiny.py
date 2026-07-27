import sys
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.base_classifier import BaseClassifier

class SwinTiny(BaseClassifier):
    """
    Swin Tiny transformer classifier subclassing BaseClassifier.
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
                weights = models.Swin_T_Weights.DEFAULT
                self.backbone = models.swin_t(weights=weights)
            except (AttributeError, TypeError):
                # Fallback to legacy pretrained flag
                self.backbone = models.swin_t(pretrained=True)
        else:
            self.backbone = models.swin_t(weights=None)
            
        # 2. Replace the classifier head
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Linear(in_features, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs the full forward pass to obtain logits.
        """
        return self.backbone(x)
        
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts visual representation features and permutes them to standard 4D format.
        Original Swin output: [batch_size, H_feat, W_feat, channels]
        Permuted output: [batch_size, channels, H_feat, W_feat]
        """
        features = self.backbone.features(x)
        # Permute to (batch_size, channels, H_feat, W_feat)
        return features.permute(0, 3, 1, 2)
