"""
src/fusion/missingness/__init__.py
Phase C11.8: Missing Modality Robustness Package
"""

from src.fusion.missingness.availability_mask import (
    ALL_REGIMES,
    NON_EMPTY_REGIMES,
    BIMODAL_REGIMES,
    UNIMODAL_REGIMES,
    MODALITY_NAMES,
    apply_availability_mask,
    verify_availability_invariants,
)
from src.fusion.missingness.missingness_result import (
    RegimeEvaluation,
    RedistributionRecord,
    StressScenarioRecord,
    MaskedInvarianceRecord,
)
from src.fusion.missingness.robustness_metrics import (
    calc_stats,
    compute_risk_delta,
    compute_dcri_delta,
    compute_authority_redistribution,
    compute_dcri_decomposition,
    compute_routing_entropy,
    compute_max_authority,
    compute_bootstrap_ci,
)
from src.fusion.missingness.dropout_scenarios import (
    get_regime_packet,
    generate_sequential_ladders,
    get_stress_packet,
    get_random_dropout_packet,
)
from src.fusion.missingness.stress_tests import (
    evaluate_masked_value_invariance,
    evaluate_stress_scenarios,
    evaluate_unavailable_vs_low_quality,
)
from src.fusion.missingness.explainability import generate_missingness_diagnostic
from src.fusion.missingness.missingness_engine import MissingnessEngine

__all__ = [
    "ALL_REGIMES",
    "NON_EMPTY_REGIMES",
    "BIMODAL_REGIMES",
    "UNIMODAL_REGIMES",
    "MODALITY_NAMES",
    "apply_availability_mask",
    "verify_availability_invariants",
    "RegimeEvaluation",
    "RedistributionRecord",
    "StressScenarioRecord",
    "MaskedInvarianceRecord",
    "calc_stats",
    "compute_risk_delta",
    "compute_dcri_delta",
    "compute_authority_redistribution",
    "compute_dcri_decomposition",
    "compute_routing_entropy",
    "compute_max_authority",
    "compute_bootstrap_ci",
    "get_regime_packet",
    "generate_sequential_ladders",
    "get_stress_packet",
    "get_random_dropout_packet",
    "evaluate_masked_value_invariance",
    "evaluate_stress_scenarios",
    "evaluate_unavailable_vs_low_quality",
    "generate_missingness_diagnostic",
    "MissingnessEngine",
]
