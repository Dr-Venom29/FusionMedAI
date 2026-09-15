import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Optional, Tuple

WAGNER_CLASSES = {
    0: "Grade 1 (Superficial Ulcer)",
    1: "Grade 2 (Deep Ulcer)",
    2: "Grade 3 (Abscess / Osteomyelitis)",
    3: "Grade 4 (Gangrene)"
}

def overlay_heatmap(
    img: np.ndarray,
    cam: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Overlays normalized Grad-CAM heatmap on RGB image.
    
    Args:
        img: RGB image array of shape (H, W, 3), range [0, 1] or [0, 255]
        cam: 2D CAM array of shape (H, W), range [0, 1]
        colormap: OpenCV colormap enum (default COLORMAP_JET)
        alpha: Blending ratio (default 0.5)
        
    Returns:
        Tuple of (overlay_rgb, heatmap_rgb) in range [0, 1]
    """
    if img.max() > 1.0:
        img_norm = img.astype(np.float32) / 255.0
    else:
        img_norm = img.copy().astype(np.float32)
        
    cam_uint8 = np.uint8(255 * np.clip(cam, 0, 1))
    heatmap_bgr = cv2.applyColorMap(cam_uint8, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    overlay = (1.0 - alpha) * img_norm + alpha * heatmap_rgb
    overlay = np.clip(overlay, 0.0, 1.0)
    return overlay, heatmap_rgb

def render_explanation_panel(
    img: np.ndarray,
    cam: np.ndarray,
    pred_class: int,
    true_class: Optional[int] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Renders 3-panel figure: Original Image | Heatmap | Grad-CAM Overlay.
    """
    overlay, heatmap = overlay_heatmap(img, cam)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img if img.max() <= 1.0 else img / 255.0)
    axes[0].set_title("Original RGB Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")
    
    axes[1].imshow(heatmap)
    axes[1].set_title("Grad-CAM Heatmap", fontsize=12, fontweight="bold")
    axes[1].axis("off")
    
    axes[2].imshow(overlay)
    pred_str = WAGNER_CLASSES.get(pred_class, f"Class {pred_class}")
    title_str = f"Overlay — Pred: {pred_str}"
    if true_class is not None:
        true_str = WAGNER_CLASSES.get(true_class, f"Class {true_class}")
        title_str += f"\nTrue: {true_str}"
    axes[2].set_title(title_str, fontsize=11, fontweight="bold")
    axes[2].axis("off")
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
