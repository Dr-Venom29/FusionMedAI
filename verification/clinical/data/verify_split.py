"""
FusionMedAI - Phase C2 Split Verification Check
"""
import sys
import os
from verify_pipeline import run_c2_verification

if __name__ == "__main__":
    success = run_c2_verification()
    sys.exit(0 if success else 1)
