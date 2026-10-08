"""
src/fusion/degradation/__init__.py
Phase C11.10: Input Degradation Benchmark & Quality-Aware Robustness Package
"""

from src.fusion.degradation.degradation_spec import (
    SEVERITY_D0_CLEAN,
    SEVERITY_D1_MILD,
    SEVERITY_D2_MODERATE,
    SEVERITY_D3_SEVERE,
    SEVERITY_LEVELS,
    MODALITY_RETINA,
    MODALITY_FOOT,
    MODALITY_CLINICAL,
    ALL_MODALITIES,
    RETINA_OPERATORS,
    FOOT_OPERATORS,
    CLINICAL_OPERATORS,
    ALL_SCENARIOS,
    get_operator_params,
    deterministic_sample_seed,
)

from src.fusion.degradation.image_degradation import (
    apply_image_degradation,
    compute_degraded_image_quality,
    apply_image_degradation_to_record,
)
from src.fusion.degradation.clinical_degradation import (
    apply_clinical_degradation,
    compute_degraded_clinical_quality,
    apply_clinical_degradation_to_record,
)
from src.fusion.degradation.degradation_result import (
    QualityResponseRecord,
    RoutingResponseRecord,
    SlopeRecord,
    MonotonicityRecord,
    BaselineComparisonRecord,
)
from src.fusion.degradation.response_metrics import (
    compute_bootstrap_ci_1d,
    compute_paired_bootstrap_ci_diff,
    compute_quality_authority_slope,
    compute_spearman_alignment,
    compute_monotonicity_rates,
    compute_routing_entropy,
    compute_conflict_metrics,
    check_hard_mask_invariance,
)
from src.fusion.degradation.degradation_engine import (
    DegradationEngine,
    apply_degradation_to_packet,
    apply_scenario_degradation,
)

__all__ = [
    "SEVERITY_D0_CLEAN",
    "SEVERITY_D1_MILD",
    "SEVERITY_D2_MODERATE",
    "SEVERITY_D3_SEVERE",
    "SEVERITY_LEVELS",
    "MODALITY_RETINA",
    "MODALITY_FOOT",
    "MODALITY_CLINICAL",
    "ALL_MODALITIES",
    "RETINA_OPERATORS",
    "FOOT_OPERATORS",
    "CLINICAL_OPERATORS",
    "ALL_SCENARIOS",
    "get_operator_params",
    "apply_image_degradation",
    "compute_degraded_image_quality",
    "apply_image_degradation_to_record",
    "apply_clinical_degradation",
    "compute_degraded_clinical_quality",
    "apply_clinical_degradation_to_record",
    "QualityResponseRecord",
    "RoutingResponseRecord",
    "SlopeRecord",
    "MonotonicityRecord",
    "BaselineComparisonRecord",
    "compute_bootstrap_ci_1d",
    "compute_paired_bootstrap_ci_diff",
    "compute_quality_authority_slope",
    "compute_spearman_alignment",
    "compute_monotonicity_rates",
    "compute_routing_entropy",
    "compute_conflict_metrics",
    "check_hard_mask_invariance",
    "DegradationEngine",
    "apply_degradation_to_packet",
    "apply_scenario_degradation",
]
