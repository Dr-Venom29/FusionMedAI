import sys
from pathlib import Path
import torch

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.model_factory import load_model
from src.models.base_classifier import BaseClassifier
import src.config as config

def verify_model():
    print("\n==================================================")
    print("Verification: Swin-Tiny Load and Forward Pass")
    print("==================================================")
    
    # 1. Instantiate the model
    print("Step A")
    print("Loading model 'swin_tiny' from model factory...")
    model = load_model(
        name="swin_tiny",
        num_classes=config.NUM_CLASSES,
        pretrained=True
    )
    print("Step B")
    
    # 2. Check inheritance
    assert isinstance(model, BaseClassifier), "Model must inherit from BaseClassifier"
    print("[OK] Model subclass validation")
    
    # 3. Print model profile
    profile = model.get_num_parameters()
    print("Step C")
    print(f"[OK] Total Parameters: {profile['total']:,}")
    print(f"[OK] Trainable Parameters: {profile['trainable']:,}")
    print(f"[OK] Model Size: {profile['size_mb']:.2f} MB")
    
    # 4. Device mapping
    device = config.DEVICE
    print(f"Mapping model to device: '{device}'...")
    model = model.to(device)
    print("Step D")
    
    # 5. Forward pass verification
    dummy_input = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).to(device)
    print("Step E")
    print(f"Running dummy forward pass with input shape: {list(dummy_input.shape)}...")
    
    model.eval()
    print("Step F")
    with torch.no_grad():
        logits = model(dummy_input)
    print("Step G")
        
    print(f"[OK] Output shape: {list(logits.shape)}")
    assert logits.shape == (2, config.NUM_CLASSES), f"Expected shape [2, {config.NUM_CLASSES}], got {list(logits.shape)}"
    print("[OK] Logits shape assertion")
    
    # 6. Feature extraction verification
    print("Testing visual feature extraction...")
    with torch.no_grad():
        features = model.extract_features(dummy_input)
    print(f"[OK] Feature extraction output shape: {list(features.shape)}")
    assert features.ndim == 4, "Expected a 4D feature map (Batch, Channels, Height, Width)"
    print("[OK] Feature extraction shape assertion")
    
    # 7. Mixed precision check
    print("Testing forward pass under Mixed Precision (AMP)...")
    if device == "cuda":
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                logits_amp = model(dummy_input)
        print("[OK] AMP forward pass assertion")
    else:
        print("[Skipped] AMP check skipped on CPU device")
        
    print("\n=== SWIN-TINY VERIFICATION SUCCESSFUL ===")
    print("==================================================\n")

if __name__ == "__main__":
    verify_model()
