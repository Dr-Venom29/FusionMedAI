import numpy as np
import torch
import torch.nn.functional as F
import cv2

class BaseCAM:
    """
    Base class for Class Activation Mapping methods.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.model.eval()
        self.activations = None
        self.gradients = None
        self.handlers = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        # For PyTorch >= 1.8, register_full_backward_hook is preferred
        if hasattr(self.target_layer, 'register_full_backward_hook'):
            def backward_hook(module, grad_input, grad_output):
                self.gradients = grad_output[0]
            self.handlers.append(self.target_layer.register_full_backward_hook(backward_hook))
        else:
            def backward_hook(module, grad_input, grad_output):
                self.gradients = grad_output[0]
            self.handlers.append(self.target_layer.register_backward_hook(backward_hook))

        self.handlers.append(self.target_layer.register_forward_hook(forward_hook))

    def remove_hooks(self):
        for handler in self.handlers:
            handler.remove()
        self.handlers.clear()
    def __del__(self):
        self.remove_hooks()

    def forward(self, input_tensor, class_idx=None):
        """
        Forward pass and backpropagation to get CAM.
        """
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        score = output[:, class_idx]
        score.backward(retain_graph=True)
        
        return self._generate_cam(input_tensor.shape[2:])

    def _generate_cam(self, target_size):
        raise NotImplementedError

    @staticmethod
    def normalize_cam(cam):
        """Normalize CAM to [0, 1] range."""
        cam = cam - np.min(cam)
        cam_max = np.max(cam)
        if cam_max != 0:
            cam = cam / cam_max
        return cam

    @staticmethod
    def overlay_cam(img, cam, colormap=cv2.COLORMAP_JET, alpha=0.5):
        """
        Overlay normalized CAM onto original image.
        img: np.ndarray (H, W, 3) float [0, 1]
        cam: np.ndarray (H, W) float [0, 1]
        """
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), colormap)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[..., ::-1] # BGR to RGB
        
        overlay = (1 - alpha) * img + alpha * heatmap
        overlay = np.clip(overlay, 0, 1)
        return overlay, heatmap
