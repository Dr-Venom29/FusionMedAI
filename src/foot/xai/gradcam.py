import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from typing import Optional, Tuple, Dict, Union
from src.foot.models.factory import load_foot_final_model, build_foot_final_model

class FootGradCAM:
    """
    Grad-CAM implementation for Foot Ulcer Classification (EfficientNet-B3 & PyTorch classifiers).
    
    Target layer default for EfficientNet-B3: 'backbone.features.8'
    """
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        target_layer_name: str = "backbone.features.8",
        device: str = "cpu"
    ):
        self.device = torch.device(device) if isinstance(device, str) else device
        if model is None:
            self.model = load_foot_final_model(device=str(self.device))
        else:
            self.model = model.to(self.device)
            
        self.model.eval()
        self.target_layer_name = target_layer_name
        self.target_layer = self._find_target_layer(target_layer_name)
        
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.handlers = []
        self._register_hooks()

    def _find_target_layer(self, layer_name: str) -> nn.Module:
        """Resolves nested layer by string name e.g. 'backbone.features.8'."""
        parts = layer_name.split(".")
        curr = self.model
        for part in parts:
            if part.isdigit():
                curr = curr[int(part)]
            elif hasattr(curr, part):
                curr = getattr(curr, part)
            else:
                raise ValueError(f"Could not find layer module part '{part}' in '{layer_name}'.")
        return curr

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        if hasattr(self.target_layer, "register_full_backward_hook"):
            self.handlers.append(self.target_layer.register_full_backward_hook(backward_hook))
        else:
            self.handlers.append(self.target_layer.register_backward_hook(backward_hook))

        self.handlers.append(self.target_layer.register_forward_hook(forward_hook))

    def remove_hooks(self):
        for handler in self.handlers:
            handler.remove()
        self.handlers.clear()

    def __del__(self):
        self.remove_hooks()

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
        target_size: Tuple[int, int] = (224, 224)
    ) -> Tuple[np.ndarray, int]:
        """
        Computes Grad-CAM spatial attribution map for input_tensor [1, 3, H, W].
        
        Returns:
            Tuple of (cam_normalized_numpy_2d, predicted_or_target_class_idx)
        """
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
            
        input_tensor = input_tensor.to(self.device).requires_grad_(True)
        self.model.zero_grad()
        
        output = self.model(input_tensor)
        if isinstance(output, dict):
            logits = output["logits"]
        else:
            logits = output
            
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
            
        score = logits[0, class_idx]
        score.backward(retain_graph=False)
        
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations were not captured by hooks.")
            
        # Global Average Pooling of gradients across spatial dimensions (dim 2, 3)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of feature activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
        # Apply ReLU activation to keep positive attributions
        cam = F.relu(cam)
        
        cam_np = cam.squeeze().cpu().numpy()
        
        # Apply slight Gaussian blur for smooth visual interpretation
        if cam_np.ndim == 2 and cam_np.shape[0] > 1 and cam_np.shape[1] > 1:
            cam_np = cv2.GaussianBlur(cam_np, (3, 3), 0)
            
        # Min-max normalize map to [0, 1]
        cam_min = np.min(cam_np)
        cam_max = np.max(cam_np)
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)
            
        # Resize spatial map to target image dimensions
        cam_resized = cv2.resize(cam_np, (target_size[1], target_size[0]))
        return cam_resized, class_idx
