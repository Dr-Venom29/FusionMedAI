"""
FusionMedAI - Phase C11.4: Dynamic Router Package (ACARA-U v2)
Exposes public router classes, input containers, and coefficients.
"""

from src.fusion.router.router_input import (
    ModalityChannelInput,
    RouterInput,
    RouterContractValidationError,
    VALID_MODALITIES,
    FROZEN_RELIABILITY_MAP,
)
from src.fusion.router.coefficients import (
    RouterCoefficients,
    RouterCoefficientError,
)
from src.fusion.router.router_result import RouterResult
from src.fusion.router.acarau_router import ACARAUv2Router

__all__ = [
    "ModalityChannelInput",
    "RouterInput",
    "RouterContractValidationError",
    "VALID_MODALITIES",
    "FROZEN_RELIABILITY_MAP",
    "RouterCoefficients",
    "RouterCoefficientError",
    "RouterResult",
    "ACARAUv2Router",
]
