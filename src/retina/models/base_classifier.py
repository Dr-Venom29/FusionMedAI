import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseClassifier(nn.Module, ABC):
    """
    Abstract base class for all image classifier models in the FusionMedAI framework.
    Ensures a consistent interface for feature extraction, head replacement, and counting parameters.
    """
    
    def __init__(self) -> None:
        super().__init__()
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass.
        
        Args:
            x: Input image tensor of shape [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Class logits of shape [batch_size, num_classes]
        """
        pass
        
    @abstractmethod
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts feature representations before the classification head.
        Useful for transfer learning, multimodal fusion, and visualizations (e.g., Grad-CAM).
        
        Args:
            x: Input image tensor.
            
        Returns:
            torch.Tensor: Feature tensor.
        """
        pass

    def get_num_parameters(self) -> dict:
        """
        Returns the parameter count details of the model.
        
        Returns:
            dict: Dictionary with 'total' and 'trainable' parameter counts, and model size in MB.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Estimate size in MB (assuming float32: 4 bytes per parameter)
        model_size_mb = (total_params * 4) / (1024 * 1024)
        
        return {
            "total": total_params,
            "trainable": trainable_params,
            "size_mb": model_size_mb
        }
