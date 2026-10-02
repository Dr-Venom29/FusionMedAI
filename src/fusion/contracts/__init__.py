"""
FusionMedAI - Unified Modality Output Contracts and Adapters (Phase C11.1).
"""

from src.fusion.contracts.modality_output import (
    ModalityOutput,
    ModalityContractValidationError,
    project_retina_risk,
    project_foot_risk,
    project_clinical_risk,
)
from src.fusion.contracts.retina_adapter import RetinaAdapter
from src.fusion.contracts.foot_adapter import FootAdapter
from src.fusion.contracts.clinical_adapter import ClinicalAdapter

__all__ = [
    "ModalityOutput",
    "ModalityContractValidationError",
    "project_retina_risk",
    "project_foot_risk",
    "project_clinical_risk",
    "RetinaAdapter",
    "FootAdapter",
    "ClinicalAdapter",
]
