import torch
import torch.optim as optim
from typing import Iterator

def get_optimizer(
    model_params: Iterator[torch.nn.Parameter],
    opt_type: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 1e-4
) -> optim.Optimizer:
    """
    Constructs and returns a PyTorch optimizer.
    
    Args:
        model_params: Iterator of parameters to optimize.
        opt_type: Optimizer type ('adamw', 'adam', 'sgd').
        lr: Base learning rate.
        weight_decay: Weight decay penalty.
        
    Returns:
        optim.Optimizer: Configured optimizer instance.
        
    Raises:
        ValueError: If opt_type is unsupported.
    """
    opt_type_lower = opt_type.lower()
    
    if opt_type_lower == "adamw":
        return optim.AdamW(model_params, lr=lr, weight_decay=weight_decay)
    elif opt_type_lower == "adam":
        return optim.Adam(model_params, lr=lr, weight_decay=weight_decay)
    elif opt_type_lower == "sgd":
        return optim.SGD(model_params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(
            f"Unsupported optimizer type: '{opt_type}'. "
            f"Currently registered optimizers: ['adamw', 'adam', 'sgd']"
        )
