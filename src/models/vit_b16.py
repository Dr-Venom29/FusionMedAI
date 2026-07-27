import sys
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.base_classifier import BaseClassifier

class ViTB16(BaseClassifier):
    """
    Vision Transformer (ViT-B/16) classifier subclassing BaseClassifier.
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
                weights = models.ViT_B_16_Weights.DEFAULT
                self.backbone = models.vit_b_16(weights=weights)
            except (AttributeError, TypeError):
                # Fallback to legacy pretrained flag
                self.backbone = models.vit_b_16(pretrained=True)
        else:
            self.backbone = models.vit_b_16(weights=None)
            
        # 2. Replace the classification head
        in_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Linear(in_features, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs the full forward pass to obtain logits.
        """
        return self.backbone(x)
        
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts patch embeddings from the transformer and reshapes to standard 4D format.
        Original encoder output: [batch_size, 197, 768] (class token + 196 patch tokens)
        Reshaped patch tokens output: [batch_size, 768, 14, 14]
        """
        n = x.shape[0]
        x_proc = self.backbone._process_input(x)
        batch_class_token = self.backbone.class_token.expand(n, -1, -1)
        x_concat = torch.cat([batch_class_token, x_proc], dim=1)
        encoder_output = self.backbone.encoder(x_concat)
        
        # encoder_output[:, 1:] contains the 196 patch tokens, shape (batch_size, 196, 768)
        patch_tokens = encoder_output[:, 1:]
        # Reshape to spatial grid: (batch_size, 14, 14, 768) and permute to (batch_size, 768, 14, 14)
        feature_map = patch_tokens.reshape(n, 14, 14, 768).permute(0, 3, 1, 2)
        return feature_map
