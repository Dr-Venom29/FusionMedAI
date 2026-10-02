"""
Modality Reliability Package (Phase C11.3).
"""

from src.fusion.reliability.reliability_result import (
    ModalityReliability,
    GlobalReliabilitySnapshot,
    ReliabilityContractError,
)
from src.fusion.reliability.auc import (
    compute_binary_auc,
    compute_macro_ovr_auc,
)
from src.fusion.reliability.ece import (
    compute_equal_frequency_ece,
)
from src.fusion.reliability.global_reliability import (
    FROZEN_RETINA_AUC,
    FROZEN_RETINA_ECE,
    FROZEN_RETINA_RELIABILITY,
    FROZEN_FOOT_AUC,
    FROZEN_FOOT_ECE,
    FROZEN_FOOT_RELIABILITY,
    FROZEN_CLINICAL_AUC,
    FROZEN_CLINICAL_ECE,
    FROZEN_CLINICAL_RELIABILITY,
    get_frozen_retina_reliability,
    get_frozen_foot_reliability,
    get_frozen_clinical_reliability,
    get_global_reliability_snapshot,
)

__all__ = [
    "ModalityReliability",
    "GlobalReliabilitySnapshot",
    "ReliabilityContractError",
    "compute_binary_auc",
    "compute_macro_ovr_auc",
    "compute_equal_frequency_ece",
    "FROZEN_RETINA_AUC",
    "FROZEN_RETINA_ECE",
    "FROZEN_RETINA_RELIABILITY",
    "FROZEN_FOOT_AUC",
    "FROZEN_FOOT_ECE",
    "FROZEN_FOOT_RELIABILITY",
    "FROZEN_CLINICAL_AUC",
    "FROZEN_CLINICAL_ECE",
    "FROZEN_CLINICAL_RELIABILITY",
    "get_frozen_retina_reliability",
    "get_frozen_foot_reliability",
    "get_frozen_clinical_reliability",
    "get_global_reliability_snapshot",
]
