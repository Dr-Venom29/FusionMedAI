import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any, Union
from src.foot.models.base_classifier import FootBaseClassifier

class FootResNet50(FootBaseClassifier):
    """
    ResNet-50 Baseline Architecture for Foot DFU Wagner 4-Class Classification.
    Recommended baseline reference model (Phase 10.4.4).
    """
    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        dropout_rate: float = 0.2
    ):
        super().__init__(num_classes=num_classes)
        
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.backbone(x)
        return self.format_output(logits)
