"""
FusionMedAI - Phase C11.13: DCRI Global Uncertainty Penalty Selection Module
"""

from .selection_config import (
    ALPHA_REF,
    BETA_REF,
    GAMMA_REF,
    ETA_REF,
    DELTA_PROVISIONAL_HISTORICAL,
    CANDIDATE_DELTA_GRID,
    SEED,
    N_PACKETS,
    N_BOOTSTRAPS,
    ACTIVE_REGIMES,
    ALL_REGIMES,
)
from .candidate_grid import (
    DeltaCandidateItem,
    get_candidate_grid,
    validate_candidate_grid,
)
from .selection_evaluator import SelectionEvaluator
from .regime_evaluator import RegimeEvaluator
from .paired_bootstrap import PairedBootstrapComparator
from .selection_decision import SelectionDecisionEngine
from .selection_runner import DeltaSelectionExperimentRunner

__all__ = [
    "ALPHA_REF",
    "BETA_REF",
    "GAMMA_REF",
    "ETA_REF",
    "DELTA_PROVISIONAL_HISTORICAL",
    "CANDIDATE_DELTA_GRID",
    "SEED",
    "N_PACKETS",
    "N_BOOTSTRAPS",
    "ACTIVE_REGIMES",
    "ALL_REGIMES",
    "DeltaCandidateItem",
    "get_candidate_grid",
    "validate_candidate_grid",
    "SelectionEvaluator",
    "RegimeEvaluator",
    "PairedBootstrapComparator",
    "SelectionDecisionEngine",
    "DeltaSelectionExperimentRunner",
]
