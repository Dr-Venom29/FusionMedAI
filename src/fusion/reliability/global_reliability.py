"""
Global Modality Reliability Engine & Frozen Constants (Phase C11.3).
Locks empirical reliability priors R_i = 0.5 * (AUC_i + (1 - ECE_i)) strictly derived from validation splits.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import json

from src.fusion.reliability.reliability_result import (
    ModalityReliability,
    GlobalReliabilitySnapshot,
)

# =========================================================================
# Frozen Empirical Validation Reliability Constants (Phase C11.3 Sealed)
# =========================================================================

# 1. Retina (EfficientNet-B3 + Temperature Scaling, N_val=366)
FROZEN_RETINA_AUC: float = 0.911149
FROZEN_RETINA_ECE: float = 0.051237
FROZEN_RETINA_RELIABILITY: float = 0.929956

# 2. Foot (EfficientNet-B3 + Vector Scaling, N_val=1006)
FROZEN_FOOT_AUC: float = 0.884435
FROZEN_FOOT_ECE: float = 0.039903
FROZEN_FOOT_RELIABILITY: float = 0.922266

# 3. Clinical (CatBoost HPO + Isotonic Calibration, N_val=14911)
FROZEN_CLINICAL_AUC: float = 0.650764
FROZEN_CLINICAL_ECE: float = 0.000000
FROZEN_CLINICAL_RELIABILITY: float = 0.825382


def get_frozen_retina_reliability() -> ModalityReliability:
    """Returns the immutable ModalityReliability contract object for Retina."""
    return ModalityReliability(
        modality="retina",
        auc=FROZEN_RETINA_AUC,
        ece=FROZEN_RETINA_ECE,
        reliability=FROZEN_RETINA_RELIABILITY,
        n_val_samples=366,
        details={
            "architecture": "EfficientNet-B3",
            "calibration_method": "Temperature_Scaling",
            "auc_type": "Macro_OvR_5_Class",
            "ece_bins": 10,
            "ece_binning": "equal_frequency",
        }
    )


def get_frozen_foot_reliability() -> ModalityReliability:
    """Returns the immutable ModalityReliability contract object for Foot."""
    return ModalityReliability(
        modality="foot",
        auc=FROZEN_FOOT_AUC,
        ece=FROZEN_FOOT_ECE,
        reliability=FROZEN_FOOT_RELIABILITY,
        n_val_samples=1006,
        details={
            "architecture": "EfficientNet-B3",
            "calibration_method": "Vector_Scaling",
            "auc_type": "Macro_OvR_4_Class",
            "ece_bins": 10,
            "ece_binning": "equal_frequency",
        }
    )


def get_frozen_clinical_reliability() -> ModalityReliability:
    """Returns the immutable ModalityReliability contract object for Clinical."""
    return ModalityReliability(
        modality="clinical",
        auc=FROZEN_CLINICAL_AUC,
        ece=FROZEN_CLINICAL_ECE,
        reliability=FROZEN_CLINICAL_RELIABILITY,
        n_val_samples=14911,
        details={
            "architecture": "CatBoost_HPO",
            "calibration_method": "Isotonic_Calibration",
            "auc_type": "Binary_ROC_AUC",
            "ece_bins": 10,
            "ece_binning": "equal_frequency",
        }
    )


def get_global_reliability_snapshot() -> GlobalReliabilitySnapshot:
    """
    Returns the complete immutable container of global reliability coefficients
    ready for downstream consumption by the ACARA-U router (C11.4).
    """
    return GlobalReliabilitySnapshot(
        retina=get_frozen_retina_reliability(),
        foot=get_frozen_foot_reliability(),
        clinical=get_frozen_clinical_reliability(),
        method="uniform_auc_calibration_reliability",
        data_source="locked_validation_only",
        test_used_for_selection=False,
    )
