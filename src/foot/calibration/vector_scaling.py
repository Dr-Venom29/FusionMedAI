import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple

class FootVectorScaler(nn.Module):
    """
    Vector Scaling for Foot Ulcer 4-class Wagner probabilities.
    
    Formulation:
        z' = w * z + b   (element-wise scaling per class channel)
        p  = softmax(z')
        
    Parameters:
        weights: nn.Parameter of shape [num_classes] (initialized to 1.0)
        bias: nn.Parameter of shape [num_classes] (initialized to 0.0)
    """
    def __init__(self, num_classes: int = 4):
        super(FootVectorScaler, self).__init__()
        self.num_classes = num_classes
        self.weights = nn.Parameter(torch.ones(num_classes))
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return self.scale_logits(logits)

    def scale_logits(self, logits: torch.Tensor) -> torch.Tensor:
        w = self.weights.to(logits.device)
        b = self.bias.to(logits.device)
        return logits * w + b

    def fit(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Fits optimal scaling weights w* and bias b* exclusively on validation logits minimizing Cross-Entropy (NLL).
        """
        val_logits_dev = val_logits.clone().detach()
        val_labels_dev = val_labels.clone().detach()
        
        optimizer = optim.LBFGS([self.weights, self.bias], lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()
        
        def closure():
            optimizer.zero_grad()
            scaled_logits = self.scale_logits(val_logits_dev)
            loss = criterion(scaled_logits, val_labels_dev)
            loss.backward()
            return loss
            
        optimizer.step(closure)
        
        return {
            "weights": self.weights.detach().cpu().numpy(),
            "bias": self.bias.detach().cpu().numpy()
        }
