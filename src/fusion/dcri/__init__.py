"""
FusionMedAI - Phase C11.6: DCRI Risk Aggregation Package
Implements uncertainty-aware risk aggregation, uncertainty burden estimation, and DCRI decomposition.
"""

from src.fusion.dcri.dcri_result import (
    DCRIResult,
    DCRIContractError,
)
from src.fusion.dcri.aggregation import (
    compute_weighted_risk_contributions,
    compute_r_fusion,
)
from src.fusion.dcri.uncertainty_penalty import (
    compute_uncertainty_burden,
    compute_uncertainty_penalty_contributions,
    compute_uncertainty_penalty,
)
from src.fusion.dcri.dcri_engine import (
    DCRIEngine,
)

__all__ = [
    "DCRIResult",
    "DCRIContractError",
    "compute_weighted_risk_contributions",
    "compute_r_fusion",
    "compute_uncertainty_burden",
    "compute_uncertainty_penalty_contributions",
    "compute_uncertainty_penalty",
    "DCRIEngine",
]
