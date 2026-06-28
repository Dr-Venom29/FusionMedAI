import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Any
from src.utils.average_meter import AverageMeter
from src.training.metrics import calculate_metrics

def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str = "cpu"
) -> Tuple[float, Dict[str, Any]]:
    """
    Performs validation evaluation over the dataset loader.
    
    Args:
        model: PyTorch classification model.
        loader: Validation DataLoader.
        criterion: Loss function module.
        device: Device execution target ('cpu' or 'cuda').
        
    Returns:
        Tuple[float, Dict[str, Any]]: (average_loss, metrics_dictionary)
    """
    model.eval()
    
    loss_meter = AverageMeter("Loss")
    
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss_meter.update(loss.item(), inputs.size(0))
            
            _, preds = torch.max(outputs, 1)
            
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    
    # Calculate validation metrics (y_probs is None to speed up epoch valuation)
    metrics = calculate_metrics(y_true=all_targets, y_pred=all_preds, y_probs=None)
    
    return loss_meter.avg, metrics
