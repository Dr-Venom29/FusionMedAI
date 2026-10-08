"""
src/fusion/combination_analysis/__init__.py
Phase C11.9: Modality-Combination Distribution & Tail-Robustness Analysis Package
"""

from src.fusion.combination_analysis.combination_definition import (
    ALL_COMBINATIONS,
    NON_EMPTY_COMBINATIONS,
    BIMODAL_COMBINATIONS,
    UNIMODAL_COMBINATIONS,
    COMBINATION_RFC,
    COMBINATION_RF,
    COMBINATION_RC,
    COMBINATION_FC,
    COMBINATION_R,
    COMBINATION_F,
    COMBINATION_C,
    COMBINATION_EMPTY,
    COMBINATION_MASKS,
    COMBINATION_CARDINALITY,
    ModalityCombinationSpec,
    get_combination_spec,
    apply_combination_to_packet,
)
from src.fusion.combination_analysis.distribution_generator import (
    ALL_DISTRIBUTIONS,
    DISTRIBUTION_D1_BALANCED,
    DISTRIBUTION_D2_MODERATE_HEAD_TAIL,
    DISTRIBUTION_D3_STRONG_LONG_TAIL,
    DistributionConfig,
    AssignedPacket,
    get_distribution_config,
    generate_stratified_distribution_cohort,
)
from src.fusion.combination_analysis.head_tail_classifier import (
    TIER_HEAD,
    TIER_MIDDLE,
    TIER_TAIL,
    TIER_UNIFORM,
    ALL_TIERS,
    CombinationTierRecord,
    DistributionTierAssignment,
    classify_distribution_tiers,
)
from src.fusion.combination_analysis.combination_metrics import (
    calc_distribution_stats,
    compute_routing_entropy,
    compute_conflict_metrics,
    MIN_SAMPLE_SIZE,
)
from src.fusion.combination_analysis.tail_robustness import (
    TierSummaryRecord,
    TailRobustnessComparison,
    compute_tier_summary,
    compute_tail_robustness_comparison,
)
from src.fusion.combination_analysis.baseline_evaluator import (
    BASELINE_IDS,
    BASELINE_NAMES,
    evaluate_baselines_on_combinations,
)
from src.fusion.combination_analysis.bootstrap_analysis import (
    compute_bootstrap_ci_1d,
    compute_bootstrap_ci_std,
)
from src.fusion.combination_analysis.calibration_analysis import (
    analyze_modality_calibration_availability,
)
from src.fusion.combination_analysis.combination_result import (
    CombinationEvaluationRecord,
    DistributionEvaluationRecord,
    CrossDistributionSensitivityRecord,
)
from src.fusion.combination_analysis.combination_engine import (
    ModalityCombinationEngine,
)

__all__ = [
    "ALL_COMBINATIONS",
    "NON_EMPTY_COMBINATIONS",
    "BIMODAL_COMBINATIONS",
    "UNIMODAL_COMBINATIONS",
    "COMBINATION_RFC",
    "COMBINATION_RF",
    "COMBINATION_RC",
    "COMBINATION_FC",
    "COMBINATION_R",
    "COMBINATION_F",
    "COMBINATION_C",
    "COMBINATION_EMPTY",
    "COMBINATION_MASKS",
    "COMBINATION_CARDINALITY",
    "ModalityCombinationSpec",
    "get_combination_spec",
    "apply_combination_to_packet",
    "ALL_DISTRIBUTIONS",
    "DISTRIBUTION_D1_BALANCED",
    "DISTRIBUTION_D2_MODERATE_HEAD_TAIL",
    "DISTRIBUTION_D3_STRONG_LONG_TAIL",
    "DistributionConfig",
    "AssignedPacket",
    "get_distribution_config",
    "generate_stratified_distribution_cohort",
    "TIER_HEAD",
    "TIER_MIDDLE",
    "TIER_TAIL",
    "TIER_UNIFORM",
    "ALL_TIERS",
    "CombinationTierRecord",
    "DistributionTierAssignment",
    "classify_distribution_tiers",
    "calc_distribution_stats",
    "compute_routing_entropy",
    "compute_conflict_metrics",
    "MIN_SAMPLE_SIZE",
    "TierSummaryRecord",
    "TailRobustnessComparison",
    "compute_tier_summary",
    "compute_tail_robustness_comparison",
    "BASELINE_IDS",
    "BASELINE_NAMES",
    "evaluate_baselines_on_combinations",
    "compute_bootstrap_ci_1d",
    "compute_bootstrap_ci_std",
    "analyze_modality_calibration_availability",
    "CombinationEvaluationRecord",
    "DistributionEvaluationRecord",
    "CrossDistributionSensitivityRecord",
    "ModalityCombinationEngine",
]
