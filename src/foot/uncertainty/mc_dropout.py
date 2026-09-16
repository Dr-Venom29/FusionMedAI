import torch
import torch.nn as nn

def enable_foot_mc_dropout(model: nn.Module) -> int:
    """
    Enables Monte Carlo Dropout during inference by switching all nn.Dropout modules
    to .train() mode while strictly keeping all BatchNorm and other layers in .eval() mode.
    
    Args:
        model: PyTorch model (e.g., EfficientNet-B3 wrapper)
        
    Returns:
        int: Number of nn.Dropout modules set to training mode
    """
    # First set entire model to eval mode to ensure BatchNorm statistics remain frozen
    model.eval()
    
    # Switch only nn.Dropout modules to training mode for stochastic MC passes
    dropout_count = 0
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
            dropout_count += 1
            
    return dropout_count
