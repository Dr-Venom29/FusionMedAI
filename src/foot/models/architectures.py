import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any, Union
from src.foot.models.base_classifier import FootBaseClassifier

class FootEfficientNetB0(FootBaseClassifier):
    """EfficientNet-B0 Architecture Wrapper for Foot DFU Wagner Classification."""
    def __init__(self, num_classes: int = 4, pretrained: bool = True, dropout_rate: float = 0.2):
        super().__init__(num_classes=num_classes)
        try:
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
        except AttributeError:
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.format_output(self.backbone(x))


class FootEfficientNetB3(FootBaseClassifier):
    """EfficientNet-B3 Architecture Wrapper for Foot DFU Wagner Classification."""
    def __init__(self, num_classes: int = 4, pretrained: bool = True, dropout_rate: float = 0.2):
        super().__init__(num_classes=num_classes)
        try:
            weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b3(weights=weights)
        except AttributeError:
            self.backbone = models.efficientnet_b3(pretrained=pretrained)
            
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.format_output(self.backbone(x))


class FootConvNeXtTiny(FootBaseClassifier):
    """ConvNeXt-Tiny Architecture Wrapper for Foot DFU Wagner Classification."""
    def __init__(self, num_classes: int = 4, pretrained: bool = True, dropout_rate: float = 0.2):
        super().__init__(num_classes=num_classes)
        try:
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            self.backbone = models.convnext_tiny(weights=weights)
        except AttributeError:
            self.backbone = models.convnext_tiny(pretrained=pretrained)
            
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.format_output(self.backbone(x))


class FootSwinTiny(FootBaseClassifier):
    """Swin-Tiny Architecture Wrapper for Foot DFU Wagner Classification."""
    def __init__(self, num_classes: int = 4, pretrained: bool = True, dropout_rate: float = 0.2):
        super().__init__(num_classes=num_classes)
        try:
            weights = models.Swin_T_Weights.DEFAULT if pretrained else None
            self.backbone = models.swin_t(weights=weights)
        except AttributeError:
            self.backbone = models.swin_t(pretrained=pretrained)
            
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.format_output(self.backbone(x))


class FootViTB16(FootBaseClassifier):
    """ViT-B/16 Architecture Wrapper for Foot DFU Wagner Classification."""
    def __init__(self, num_classes: int = 4, pretrained: bool = True, dropout_rate: float = 0.2):
        super().__init__(num_classes=num_classes)
        try:
            weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
            self.backbone = models.vit_b_16(weights=weights)
        except AttributeError:
            self.backbone = models.vit_b_16(pretrained=pretrained)
            
        in_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.format_output(self.backbone(x))
