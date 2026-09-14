import torch
import torch.nn as nn
from typing import List, Optional

def get_loss_function(
    loss_type: str = "unweighted",
    class_weights: Optional[List[float]] = None,
    device: str = "cpu"
) -> nn.Module:
    """
    Returns configured CrossEntropyLoss module according to Baseline Experimental Contract (10.4.7).
    
    Args:
        loss_type: 'unweighted' (Standard Baseline CrossEntropy) or 'weighted' (Sqrt Inverse Frequency)
        class_weights: List of 4 floats representing class weights
        device: 'cpu' or 'cuda'
        
    Returns:
        torch.nn.Module loss criterion.
    """
    if loss_type == "unweighted" or class_weights is None:
        return nn.CrossEntropyLoss()
    elif loss_type == "weighted":
        weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        return nn.CrossEntropyLoss(weight=weights_tensor)
    else:
        raise ValueError(f"Unknown loss_type '{loss_type}'. Must be 'unweighted' or 'weighted'.")
