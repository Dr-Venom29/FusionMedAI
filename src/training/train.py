import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from typing import Tuple, Dict, Any, Optional
from src.utils.average_meter import AverageMeter

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    use_amp: bool = True,
    scaler: Optional[GradScaler] = None
) -> Tuple[float, float]:
    """
    Performs a single training epoch over the dataset loader.
    
    Args:
        model: PyTorch classification model.
        loader: Training DataLoader.
        criterion: Loss function module.
        optimizer: Optimizer instance.
        device: Device execution target ('cpu' or 'cuda').
        use_amp: If True, uses Automatic Mixed Precision (AMP).
        scaler: GradScaler instance for AMP (required if use_amp=True on CUDA).
        
    Returns:
        Tuple[float, float]: (average_loss, average_accuracy)
    """
    model.train()
    
    loss_meter = AverageMeter("Loss")
    correct_meter = AverageMeter("Acc")
    
    # Initialize scaler if AMP is enabled but no scaler is passed
    if use_amp and scaler is None and device == "cuda":
        scaler = GradScaler()
        
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # AMP Autocast block
        # autocast requires device_type string in newer versions, handles cpu/cuda automatically
        device_type = "cuda" if "cuda" in device else "cpu"
        
        # CPU autocast is supported in torch >= 1.10. Disable AMP on CPU if not supported.
        enable_autocast = use_amp and (device_type == "cuda")
        
        with autocast(device_type=device_type, enabled=enable_autocast):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
        if enable_autocast and scaler is not None:
            # Scale loss and perform backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        # Update metrics
        loss_meter.update(loss.item(), inputs.size(0))
        
        _, preds = torch.max(outputs, 1)
        corrects = torch.sum(preds == targets).item()
        correct_meter.update(corrects / inputs.size(0), inputs.size(0))
        
    return loss_meter.avg, correct_meter.avg
