import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple

class FootBaselineModel(nn.Module):
    """
    Primary Foot DFU Wagner 4-Class Classification Baseline Model.
    Uses pre-trained EfficientNet-B0 backbone with a custom linear classification head.
    """
    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super(FootBaselineModel, self).__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        
        if pretrained:
            weights = models.EfficientNet_B0_Weights.DEFAULT
            self.backbone = models.efficientnet_b0(weights=weights)
        else:
            self.backbone = models.efficientnet_b0(weights=None)
            
        # Replace classification head (in_features=1280 for EfficientNet-B0)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        return self.backbone(x)

def build_foot_baseline_model(num_classes: int = 4, pretrained: bool = True, device: str = "cpu") -> nn.Module:
    """Helper function to build and transfer model to specified device."""
    model = FootBaselineModel(num_classes=num_classes, pretrained=pretrained)
    return model.to(device)

if __name__ == "__main__":
    # Quick sanity check
    dummy_input = torch.randn(4, 3, 224, 224)
    model = FootBaselineModel(num_classes=4, pretrained=False)
    out = model(dummy_input)
    print(f"FootBaselineModel input shape: {dummy_input.shape} -> output shape: {out.shape}")
    assert out.shape == (4, 4), f"Expected shape (4, 4), got {out.shape}"
    print("FootBaselineModel sanity check passed!")
