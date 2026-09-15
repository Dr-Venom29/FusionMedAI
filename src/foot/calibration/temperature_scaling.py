import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple


class FootTemperatureScaler(nn.Module):
    """
    Temperature Scaling for Foot Ulcer 4-class Wagner probabilities.
    
    Formulation:
        z' = z / T
        p  = softmax(z')
        
    Parameterization:
        self.log_temperature = log(T), ensuring T > 0 strictly.
    """
    def __init__(self, initial_temperature: float = 1.0):
        super(FootTemperatureScaler, self).__init__()
        self.log_temperature = nn.Parameter(torch.ones(1) * float(np.log(initial_temperature)))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return self.scale_logits(logits)

    def scale_logits(self, logits: torch.Tensor) -> torch.Tensor:
        temp = self.temperature.to(logits.device)
        return logits / temp

    def fit(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50
    ) -> float:
        """
        Fits optimal temperature T* exclusively on validation logits minimizing Cross-Entropy (NLL).
        """
        val_logits_dev = val_logits.clone().detach()
        val_labels_dev = val_labels.clone().detach()
        
        optimizer = optim.LBFGS([self.log_temperature], lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()
        
        def closure():
            optimizer.zero_grad()
            scaled_logits = self.scale_logits(val_logits_dev)
            loss = criterion(scaled_logits, val_labels_dev)
            loss.backward()
            return loss
            
        optimizer.step(closure)
        return float(self.temperature.item())
