import numpy as np
import torch
import torch.nn.functional as F
import cv2
from typing import Tuple, Optional, Union, List, Dict

def evaluate_pointing_game(
    cam: np.ndarray,
    roi_mask_or_bbox: Union[np.ndarray, List[int], Tuple[int, int, int, int]]
) -> bool:
    """
    Evaluates Pointing Game hit criterion on 2D CAM array (H, W).
    
    Args:
        cam: 2D array of shape (H, W)
        roi_mask_or_bbox: Either binary mask (H, W) where ROI > 0, or bbox [ymin, xmin, ymax, xmax]
        
    Returns:
        bool: True if peak attribution point falls inside ROI, False otherwise.
    """
    max_idx = np.unravel_index(np.argmax(cam), cam.shape)
    peak_y, peak_x = max_idx[0], max_idx[1]
    
    if isinstance(roi_mask_or_bbox, np.ndarray):
        return bool(roi_mask_or_bbox[peak_y, peak_x] > 0)
    else:
        ymin, xmin, ymax, xmax = roi_mask_or_bbox
        return bool(ymin <= peak_y <= ymax and xmin <= peak_x <= xmax)

def evaluate_deletion_insertion(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    cam: np.ndarray,
    target_class: int,
    steps: int = 10,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Exploratory Perturbation Analysis (Deletion & Insertion AUC).
    
    Note: Deletion/Insertion AUC measures input sensitivity to top-attributed pixels under
    synthetic pixel masking. This is an exploratory feature-attribution diagnostic,
    not a formal clinical accuracy or medical segmentation gate.
    """
    model.eval()
    dev = torch.device(device)
    tensor_dev = input_tensor.to(dev)
    if tensor_dev.dim() == 3:
        tensor_dev = tensor_dev.unsqueeze(0)
        
    with torch.no_grad():
        out = model(tensor_dev)
        logits = out["logits"] if isinstance(out, dict) else out
        probs = F.softmax(logits, dim=1)
        orig_prob = probs[0, target_class].item()
        
    img_np = tensor_dev.detach().cpu().squeeze().numpy() # [3, H, W]
    
    # Compute Gaussian-blurred baseline for insertion unmasking
    blurred_np = np.zeros_like(img_np)
    for c in range(3):
        blurred_np[c] = cv2.GaussianBlur(img_np[c], (11, 11), 5.0)
        
    flat_cam = cam.flatten()
    sorted_indices = np.argsort(flat_cam)[::-1]
    total_pixels = len(sorted_indices)
    
    del_probs = [orig_prob]
    ins_probs = []
    
    # Compute initial baseline probability for insertion
    ins_baseline_tensor = torch.from_numpy(blurred_np).unsqueeze(0).to(dev)
    with torch.no_grad():
        out_base = model(ins_baseline_tensor)
        logits_base = out_base["logits"] if isinstance(out_base, dict) else out_base
        p_base = F.softmax(logits_base, dim=1)[0, target_class].item()
        ins_probs.append(p_base)
        
    for s in range(1, steps + 1):
        fraction = s / steps
        num_pixels = int(fraction * total_pixels)
        top_indices = sorted_indices[:num_pixels]
        
        # Deletion: replace top attribution pixels with blurred baseline
        del_img = img_np.copy()
        for idx in top_indices:
            c_y, c_x = np.unravel_index(idx, cam.shape)
            del_img[:, c_y, c_x] = blurred_np[:, c_y, c_x]
            
        del_tensor = torch.from_numpy(del_img).unsqueeze(0).to(dev)
        with torch.no_grad():
            out_del = model(del_tensor)
            logits_del = out_del["logits"] if isinstance(out_del, dict) else out_del
            p_del = F.softmax(logits_del, dim=1)[0, target_class].item()
            del_probs.append(p_del)
            
        # Insertion: unmask top attribution pixels onto blurred baseline
        ins_img = blurred_np.copy()
        for idx in top_indices:
            c_y, c_x = np.unravel_index(idx, cam.shape)
            ins_img[:, c_y, c_x] = img_np[:, c_y, c_x]
            
        ins_tensor = torch.from_numpy(ins_img).unsqueeze(0).to(dev)
        with torch.no_grad():
            out_ins = model(ins_tensor)
            logits_ins = out_ins["logits"] if isinstance(out_ins, dict) else out_ins
            p_ins = F.softmax(logits_ins, dim=1)[0, target_class].item()
            ins_probs.append(p_ins)
            
    # Use trapezoid integration
    if hasattr(np, "trapezoid"):
        deletion_auc = float(np.trapezoid(del_probs, dx=1.0 / steps))
        insertion_auc = float(np.trapezoid(ins_probs, dx=1.0 / steps))
    else:
        deletion_auc = float(np.trapz(del_probs, dx=1.0 / steps))
        insertion_auc = float(np.trapz(ins_probs, dx=1.0 / steps))
        
    return {
        "deletion_auc": round(deletion_auc, 4),
        "insertion_auc": round(insertion_auc, 4)
    }
