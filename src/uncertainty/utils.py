import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import torch

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))
import src.config as config

logger = logging.getLogger("Uncertainty_Utils")

def get_file_sha256(filepath: Path) -> str:
    """Compute the SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def get_git_commit_hash() -> str:
    """Returns the current git commit hash, or 'N/A' if unavailable."""
    import subprocess
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT).decode('ascii').strip()
        return commit_hash
    except Exception:
        return "N/A"

def find_latest_calibration_dir(base_dir: Path) -> Path:
    """Finds the directory matching v*_temperature_scaling with the largest version."""
    if not base_dir.exists():
        raise FileNotFoundError(f"Calibration experiments directory not found at {base_dir}")
        
    runs = []
    for path in base_dir.iterdir():
        if path.is_dir():
            match = re.match(r"^v(\d+)_temperature_scaling", path.name)
            if match:
                runs.append((int(match.group(1)), path))
                
    if not runs:
        raise FileNotFoundError(f"No temperature scaling calibration runs found in {base_dir}")
        
    # Return the path with the max version number
    runs.sort(key=lambda x: x[0])
    return runs[-1][1]

def load_and_verify_calibration(
    checkpoint_path: Path,
    calibration_dir: Optional[Path] = None,
    override_temp: Optional[float] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Loads temperature scaling factors from Step 7 experiments, verifies checkpoint integrity,
    and returns the temperature value and verification metadata.
    
    If the calibration file cannot be loaded, raises an exception (fail-fast),
    unless override_temp is specified.
    """
    verification_info = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_exists": checkpoint_path.exists(),
        "checkpoint_sha256": "N/A",
        "temperature_source": "N/A",
        "calibration_manifest_sha256": "N/A",
        "temperature_verified": False,
        "warning_issued": False,
        "warning_message": ""
    }
    
    # 1. Check checkpoint exists
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    
    checkpoint_sha256 = get_file_sha256(checkpoint_path)
    verification_info["checkpoint_sha256"] = checkpoint_sha256
    
    # 2. Handle override temperature
    if override_temp is not None:
        if override_temp <= 0:
            raise ValueError(f"Override temperature must be > 0 (got {override_temp})")
        logger.warning(f"Using MANUAL override temperature: T = {override_temp}")
        verification_info["temperature"] = override_temp
        verification_info["temperature_source"] = "manual_override"
        verification_info["warning_issued"] = True
        verification_info["warning_message"] = "Manual temperature override was provided."
        return override_temp, verification_info

    # 3. Locate calibration dir
    if calibration_dir is None:
        try:
            calibration_dir = find_latest_calibration_dir(config.CALIBRATION_EXPERIMENTS_DIR)
            logger.info(f"Automatically identified latest calibration run at: {calibration_dir}")
        except Exception as e:
            raise RuntimeError(f"Failed to find latest calibration run. Try providing explicit directory: {e}")
            
    verification_info["temperature_source"] = str(calibration_dir)
    
    # Paths to calibration files
    state_path = calibration_dir / "calibration_state.pt"
    manifest_path = calibration_dir / "manifest.json"
    
    if not state_path.exists():
        raise FileNotFoundError(
            f"Calibration state file missing at '{state_path}'. "
            f"Step 8 requires a completed calibration from Step 7. Run calibration first or use --temperature."
        )
        
    # 4. Load temperature
    try:
        state = torch.load(state_path, map_location="cpu")
        temperature = float(state["temperature"])
        verification_info["temperature"] = temperature
    except Exception as e:
        raise RuntimeError(f"Failed to load calibration state from {state_path}: {e}")
        
    if temperature <= 0:
        raise ValueError(f"Loaded temperature must be > 0, got {temperature}")
        
    # 5. Load and verify manifest
    if manifest_path.exists():
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            val_checkpoint_sha256 = manifest.get("checkpoint_sha256", "")
            verification_info["calibration_manifest_sha256"] = val_checkpoint_sha256
            
            # Check if checkpoint hash matches
            if val_checkpoint_sha256 != checkpoint_sha256:
                msg = (
                    f"Checkpoint hash mismatch! Calibration hash: {val_checkpoint_sha256}, "
                    f"Uncertainty target checkpoint hash: {checkpoint_sha256}. "
                    f"Verify you are testing the same model that was calibrated."
                )
                logger.warning(msg)
                verification_info["warning_issued"] = True
                verification_info["warning_message"] = msg
            else:
                logger.info("Integrity check passed: Checkpoint hash matches calibration validation checkpoint.")
                verification_info["temperature_verified"] = True
        except Exception as e:
            logger.warning(f"Could not parse calibration manifest at {manifest_path}: {e}")
    else:
        logger.warning(f"Calibration manifest file missing at '{manifest_path}'. Checkpoint verification skipped.")
        verification_info["warning_issued"] = True
        verification_info["warning_message"] = "Calibration manifest was missing, skipped hash checks."
        
    return temperature, verification_info
