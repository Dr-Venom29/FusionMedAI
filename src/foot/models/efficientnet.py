import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any
from src.foot.models.base_classifier import FootBaseClassifier

class FootEfficientNet(FootBaseClassifier):
    """
    EfficientNet-B0 Architecture for Foot DFU Wagner 4-Class Classification.
    Candidate baseline model.
    """
    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        dropout_rate: float = 0.2
    ):
        super().__init__(num_classes=num_classes)
        
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.backbone(x)
        return self.format_output(logits)
