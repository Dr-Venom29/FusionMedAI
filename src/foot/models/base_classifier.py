import torch
import torch.nn as nn
from typing import Dict, Any

class FootBaseClassifier(nn.Module):
    """
    Abstract Base Class for Foot DFU Wagner Classifiers.
    Standardizes output interface across backbones (ResNet, EfficientNet, ConvNeXt, Swin, ViT).
    
    Returns structured dictionary output:
    {
        "logits": Tensor of raw classification logits [B, 4],
        "probs": Tensor of softmax probabilities [B, 4],
        "predicted_class": Tensor of predicted class indices [B]
    }
    """
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes

    def format_output(self, logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        probs = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1)
        return {
            "logits": logits,
            "probs": probs,
            "predicted_class": predicted_class
        }
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Subclasses must implement forward()")
