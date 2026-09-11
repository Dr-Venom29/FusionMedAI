import torch
import torch.nn as nn
from typing import Optional

def get_loss_fn(
    loss_type: str = "ce",
    weight: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0
) -> nn.Module:
    """
    Returns the loss function module based on the requested loss_type.
    
    Args:
        loss_type: The type of loss function to load ('ce' for standard CrossEntropyLoss).
        weight: Optional tensor of class weights for handling class imbalance.
        label_smoothing: Smooths the target labels by a fraction.
        
    Returns:
        nn.Module: Configured PyTorch loss module.
        
    Raises:
        ValueError: If loss_type is unsupported.
    """
    loss_type_lower = loss_type.lower()
    
    if loss_type_lower == "ce" or loss_type_lower == "crossentropy":
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    else:
        raise ValueError(
            f"Unsupported loss function type: '{loss_type}'. "
            f"Currently registered loss types: ['ce', 'crossentropy']"
        )
