import numpy as np
import scipy.stats
import torch

def compute_entropy(probabilities):
    """
    Compute Shannon entropy of a probability distribution.
    probabilities: np.ndarray of shape (..., num_classes)
    """
    # Adding a small epsilon to avoid log(0)
    epsilon = 1e-9
    probs = np.clip(probabilities, epsilon, 1. - epsilon)
    return scipy.stats.entropy(probs, axis=-1)

def compute_margin(probabilities):
    """
    Compute margin (difference between top 2 probabilities).
    probabilities: np.ndarray of shape (N, num_classes) or (num_classes,)
    """
    if probabilities.ndim == 1:
        sorted_probs = np.sort(probabilities)[::-1]
        if len(sorted_probs) > 1:
            return sorted_probs[0] - sorted_probs[1]
        return 0.0
    
    sorted_probs = np.sort(probabilities, axis=-1)[:, ::-1]
    if sorted_probs.shape[-1] > 1:
        return sorted_probs[:, 0] - sorted_probs[:, 1]
    return np.zeros(sorted_probs.shape[0])

def get_target_layer(model, model_name):
    """
    Returns the appropriate target layer for CAM extraction based on the model architecture.
    """
    model_name = model_name.lower()
    
    if "efficientnet" in model_name:
        for name, module in model.named_modules():
            if name == "model.conv_head" or name == "conv_head":
                return module
                
    elif "convnext" in model_name:
        for name, module in model.named_modules():
            if name == "model.stages.3" or name == "stages.3":
                return module
                
    elif "swin" in model_name:
        # Note: Standard Grad-CAM is designed for convolutional feature maps.
        # Swin Transformers (and ViTs) typically require transformer-aware CAM adaptations 
        # (e.g., Attention Rollout) or reshaping of token embeddings. 
        # Support here is experimental until validated.
        for name, module in model.named_modules():
            if name == "model.norm" or name == "norm":
                return module
                
    # Fallback to the last Conv2d layer
    layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    return layers[-1] if layers else list(model.modules())[-1]

def get_git_commit_hash():
    """Returns the current git commit hash, or 'N/A' if unavailable."""
    import subprocess
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT).decode('ascii').strip()
        return commit_hash
    except Exception:
        return "N/A"
