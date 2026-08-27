import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import src.config as config
from src.retina_module import RetinaModule

def main():
    checkpoint_path = Path(config.PROJECT_ROOT) / "experiments" / "efficientnet_b3" / "checkpoints" / "best_model.pt"
    module = RetinaModule(checkpoint_path=checkpoint_path, device="cpu")
    
    # Locate test image
    img_path = Path(config.PROJECT_ROOT) / "datasets" / "raw" / "aptos2019" / "train_images" / "000c1434d8d7.png"
    assert img_path.exists(), f"Sample image not found at {img_path}"
    
    # Run prediction (25 passes for final high-quality stats matching Step 8)
    print("Running 25-pass MC Dropout prediction on sample image...")
    res = module.predict(image=img_path, mc_passes=25, generate_cam=True)
    
    # Setup examples output dir
    examples_dir = Path(config.PROJECT_ROOT) / "docs" / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    
    # Save original image
    with Image.open(img_path) as img:
        img.save(examples_dir / "retina_input.png")
        print("Saved docs/examples/retina_input.png")
        
    # Save Grad-CAM overlay
    overlay = res["cam_overlay"]
    assert overlay is not None, "Failed to generate Grad-CAM overlay."
    Image.fromarray(overlay).save(examples_dir / "retina_gradcam.png")
    print("Saved docs/examples/retina_gradcam.png")
    
    # Create the unified diagnostic output panel card
    fig = plt.figure(figsize=(15, 6.5))
    
    # Left plot: original
    ax_orig = fig.add_subplot(1, 3, 1)
    ax_orig.imshow(Image.open(img_path))
    ax_orig.set_title("Input Fundus Scan", fontweight="bold", fontsize=12, pad=10)
    ax_orig.axis("off")
    
    # Middle plot: Grad-CAM overlay
    ax_cam = fig.add_subplot(1, 3, 2)
    ax_cam.imshow(overlay)
    ax_cam.set_title("Grad-CAM Explanation", fontweight="bold", fontsize=12, pad=10)
    ax_cam.axis("off")
    
    # Right plot: Text Stats
    ax_text = fig.add_subplot(1, 3, 3)
    ax_text.axis("off")
    
    text_content = (
        f"Image ID: 000c1434d8d7.png\n\n"
        f"Predicted Diagnosis:\n"
        f"  Class {res['prediction']} ({res['prediction_label']})\n\n"
        f"Calibration & Confidence:\n"
        f"  - Raw Confidence:       {res['raw_confidence']:.2%}\n"
        f"  - Calibrated Confidence: {res['calib_confidence']:.2%}\n"
        f"  - Calibrated Entropy:    {res['calib_entropy_norm']:.4f}\n\n"
        f"MC Uncertainty (N=25):\n"
        f"  - Predictive Entropy:   {res['mc_predictive_entropy_norm']:.4f}\n"
        f"  - Predictive Variance:  {res['mc_predictive_variance']:.6f}\n"
        f"  - Mutual Information:   {res['mc_mutual_information']:.6f}\n\n"
        f"Visual Explainability:\n"
        f"  - Grad-CAM Intensity:   {res['cam_mean_intensity']:.4f}\n\n"
        f"Inference Latency:        {res['latency_ms']:.1f} ms"
    )
    
    ax_text.text(
        0.05, 0.95, text_content,
        fontsize=11, family="monospace", verticalalignment="top",
        bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=1.0)
    )
    ax_text.set_title("Unified Diagnostic Output", fontweight="bold", fontsize=12, pad=10)
    
    plt.suptitle("Retina Module Inference Example", fontweight="bold", fontsize=15, y=0.96)
    plt.tight_layout()
    plt.savefig(examples_dir / "retina_output.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved docs/examples/retina_output.png")
    print("All example assets generated successfully!")

if __name__ == "__main__":
    main()
