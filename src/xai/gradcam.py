import numpy as np
import torch
import torch.nn.functional as F
import cv2
from src.xai.cam import BaseCAM

class GradCAM(BaseCAM):
    """
    Grad-CAM implementation.
    """
    def _generate_cam(self, target_size):
        # Average pooling of the gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of forward activations
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
        # ReLU applied to the CAM
        cam = F.relu(cam)
        
        cam = cam.squeeze().cpu().detach().numpy()
        
        # Apply slight Gaussian blur for smoother heatmaps
        cam = cv2.GaussianBlur(cam, (3, 3), 0)
        
        # Resize to original image size
        cam = cv2.resize(cam, (target_size[1], target_size[0]))
        return cam
