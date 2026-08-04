import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
import time

class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature



def calibrate_model(
    model: nn.Module,
    valid_loader: torch.utils.data.DataLoader,
    output_dir: Path,
    device: str = "cuda"
) -> Tuple[ModelWithTemperature, Dict[str, Any], torch.Tensor, torch.Tensor]:
    """
    Calibrate the model, save logits/labels, and save calibration state.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Collect all raw logits and labels first
    model.eval()
    model.to(device)
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            
    raw_logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    raw_probs = F.softmax(raw_logits, dim=1)
    
    # Save raw arrays
    np.save(output_dir / "validation_logits.npy", raw_logits.numpy())
    np.save(output_dir / "validation_labels.npy", labels.numpy())
    np.save(output_dir / "validation_probabilities.npy", raw_probs.numpy())
    
    # 2. Wrap model and optimize
    scaled_model = ModelWithTemperature(model).to(device)
    
    # We do the optimization directly with the pre-computed logits to save time
    nll_criterion = nn.CrossEntropyLoss().to(device)
    raw_logits_device = raw_logits.to(device)
    labels_device = labels.to(device)
    
    optimizer = optim.LBFGS([scaled_model.temperature], lr=0.01, max_iter=50)
    
    def eval():
        optimizer.zero_grad()
        loss = nll_criterion(scaled_model.temperature_scale(raw_logits_device), labels_device)
        loss.backward()
        return loss

    optimizer.step(eval)
    
    optimal_temperature = scaled_model.temperature.item()
    
    # 3. Save states
    from .metrics import compute_all_metrics
    final_metrics = compute_all_metrics(scaled_model.temperature_scale(raw_logits_device), labels_device)
    
    calibration_state = {
        "temperature": optimal_temperature,
        "NLL": final_metrics["NLL"],
        "ECE": final_metrics["ECE"],
        "timestamp": time.time()
    }
    
    torch.save(calibration_state, output_dir / "calibration_state.pt")
    
    # We also save the ModelWithTemperature's state_dict just for the wrapper
    torch.save(scaled_model.state_dict(), output_dir / "temperature_scaling.pt")
    
    return scaled_model, calibration_state, raw_logits, labels
