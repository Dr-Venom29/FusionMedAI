import sys
from pathlib import Path
import torch

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.model_factory import load_model
from src.models.base_classifier import BaseClassifier
import src.config as config

def verify_model():
    """
    Verification script for baseline model architecture, input-output shapes,
    feature extraction functionality, and parameter counting.
    """
    print("\n==================================================")
    print("Verification: Model Load and Forward Pass")
    print("==================================================")
    
    # 1. Instantiate the model
    print(f"Loading model '{config.MODEL_NAME}' from model factory...")
    model = load_model(
        name=config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        pretrained=True
    )
    
    # 2. Check inheritance
    assert isinstance(model, BaseClassifier), "Model must inherit from BaseClassifier"
    print("[OK] Model subclass validation")
    
    # 3. Print model profile
    profile = model.get_num_parameters()
    print(f"[OK] Total Parameters: {profile['total']:,}")
    print(f"[OK] Trainable Parameters: {profile['trainable']:,}")
    print(f"[OK] Model Size: {profile['size_mb']:.2f} MB")
    
    # 4. CPU/GPU Device mapping
    device = config.DEVICE
    print(f"Mapping model to device: '{device}'...")
    model = model.to(device)
    
    # 5. Forward pass verification
    dummy_input = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).to(device)
    print(f"Running dummy forward pass with input shape: {list(dummy_input.shape)}...")
    
    model.eval()
    with torch.no_grad():
        logits = model(dummy_input)
        
    print(f"[OK] Output shape: {list(logits.shape)}")
    assert logits.shape == (2, config.NUM_CLASSES), f"Expected shape [2, {config.NUM_CLASSES}], got {list(logits.shape)}"
    print("[OK] Logits shape assertion")
    
    # 6. Feature extraction verification
    print("Testing visual feature extraction for Grad-CAM support...")
    with torch.no_grad():
        features = model.extract_features(dummy_input)
    print(f"[OK] Feature extraction output shape: {list(features.shape)}")
    assert features.ndim == 4, "Expected a 4D feature map (Batch, Channels, Height, Width)"
    print("[OK] Feature extraction shape assertion")
    
    print("\n=== MODEL VERIFICATION SUCCESSFUL ===")
    print("==================================================\n")

if __name__ == "__main__":
    verify_model()
