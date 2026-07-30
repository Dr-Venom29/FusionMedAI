import matplotlib.pyplot as plt
import numpy as np

def generate_xai_panel(original, heatmap, overlay, metadata, observation, save_path=None):
    """
    Generate a publication-quality visualization panel.
    original, heatmap, overlay: np.ndarray RGB images
    metadata: dict containing GT, Prediction, Confidence, Entropy
    observation: str from clinical interpreter
    save_path: str or Path to save the image
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    # Plot images
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    axes[1].imshow(heatmap)
    axes[1].set_title("Heatmap")
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis('off')
    
    # Add text panel at the bottom
    text_info = (
        f"Ground Truth: {metadata.get('ground_truth', 'N/A')}    "
        f"Prediction: {metadata.get('prediction', 'N/A')}    "
        f"Confidence: {metadata.get('confidence', 0.0):.4f}\n"
        f"Entropy: {metadata.get('entropy', 0.0):.4f}    "
        f"Reason: {metadata.get('selected_reason', 'N/A')}\n\n"
        f"Observation: {observation}"
    )
    
    fig.text(0.5, 0.05, text_info, ha='center', va='top', fontsize=12,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))
             
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig
