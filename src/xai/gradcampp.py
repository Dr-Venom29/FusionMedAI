import numpy as np
import torch
import torch.nn.functional as F
import cv2
from src.xai.cam import BaseCAM

class GradCAMPlusPlus(BaseCAM):
    """
    Grad-CAM++ implementation.
    """
    def _generate_cam(self, target_size):
        gradients = self.gradients
        activations = self.activations
        
        # Calculate alpha coefficients
        alpha_num = gradients.pow(2)
        alpha_denom = 2 * gradients.pow(2) + torch.sum(activations * gradients.pow(3), dim=(2,3), keepdim=True)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        
        alphas = alpha_num / alpha_denom
        
        # Calculate weights
        weights = torch.sum(alphas * F.relu(gradients), dim=(2,3), keepdim=True)
        
        # Calculate CAM
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        cam = cam.squeeze().cpu().detach().numpy()
        cam = cv2.GaussianBlur(cam, (3, 3), 0)
        cam = cv2.resize(cam, (target_size[1], target_size[0]))
        return cam
