import numpy as np
import cv2

def interpret_cam_clinical(cam):
    """
    Provide scientifically defensible clinical observations based on spatial heuristics
    of the Class Activation Map (CAM).
    cam: np.ndarray (H, W) normalized to [0, 1]
    """
    H, W = cam.shape
    
    threshold = 0.5
    mask = (cam > threshold).astype(np.uint8)
    
    if np.sum(mask) == 0:
        return "Attention dispersed uniformly or no strong focus."
        
    # Connected components
    num_labels, labels_im = cv2.connectedComponents(mask)
    num_regions = num_labels - 1
    
    # Center heuristics
    M = cv2.moments(mask)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = W // 2, H // 2
        
    rel_x = cX / W
    rel_y = cY / H
    
    area_ratio = np.sum(mask) / (H * W)
    avg_intensity = np.mean(cam[mask == 1])
    
    observations = []
    
    if num_regions > 3:
        observations.append("Attention spans multiple distinct focal regions.")
    elif num_regions == 1:
        observations.append("Attention is highly localized to a single region.")
        
    if area_ratio > 0.4:
        observations.append("Attention is widely dispersed across retinal tissue.")
    
    if 0.35 < rel_x < 0.65 and 0.35 < rel_y < 0.65:
        observations.append("Attention concentrated near the central (macular) region.")
    elif rel_x < 0.15 or rel_x > 0.85 or rel_y < 0.15 or rel_y > 0.85:
        observations.append("Attention overlaps peripheral retina or extends into image borders.")
        
    if avg_intensity > 0.85:
        observations.append("Extremely high activation intensity within focal regions.")
        
    if not observations:
        observations.append("Attention is distributed across intermediate retinal zones.")
        
    return " ".join(observations)
