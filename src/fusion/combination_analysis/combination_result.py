"""
src/fusion/combination_analysis/combination_result.py
Phase C11.9: Immutable Data Classes & Result Structures

Defines typed, immutable data models for:
1. CombinationEvaluationRecord: Full metric profile for a single combination in a distribution.
2. DistributionEvaluationRecord: Overall evaluation across all 7 non-empty combinations in a distribution.
3. CrossDistributionSensitivityRecord: Global metric responses across distributions D1, D2, and D3.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional


@dataclass(frozen=True)
class CombinationEvaluationRecord:
    """Immutable evaluation record for one combination in one distribution."""
    combination_id: str
    distribution_id: str
    n_packets: int
    frequency_probability: float
    rank: int
    tier: str
    mean_w_retina: float
    mean_w_foot: float
    mean_w_clinical: float
    mean_entropy: float
    std_entropy: float
    mean_r_fusion: float
    std_r_fusion: float
    mean_dcri: float
    std_dcri: float
    mean_u_sum: float
    std_u_sum: float
    mean_delta_max: float
    mean_delta_mean: float
    mean_sigma_w: float
    mean_delta_r: float         # Sensitivity relative to RFC
    std_delta_r: float
    ci_95_r_fusion: Tuple[float, float]
    ci_95_dcri: Tuple[float, float]
    ci_95_delta_r: Tuple[float, float]
    status: str = "SUCCESS"


@dataclass(frozen=True)
class DistributionEvaluationRecord:
    """Complete evaluation summary for a specific frequency distribution (D1, D2, or D3)."""
    distribution_id: str
    name: str
    total_packets: int
    head_to_tail_ratio: Optional[float]
    combinations: Dict[str, CombinationEvaluationRecord]
    head_summary: Dict[str, Any]
    middle_summary: Dict[str, Any]
    tail_summary: Dict[str, Any]
    tail_vs_head_contrast: Dict[str, Any]
    baseline_comparison: Dict[str, Any]
    global_mean_r_fusion: float
    global_std_r_fusion: float
    global_mean_dcri: float
    global_std_dcri: float
    global_mean_entropy: float
    global_mean_u_sum: float
    global_mean_delta_max: float


@dataclass(frozen=True)
class CrossDistributionSensitivityRecord:
    """Global metric responses across all tested distributions."""
    distributions_evaluated: List[str]
    global_r_fusion_by_dist: Dict[str, float]
    global_dcri_by_dist: Dict[str, float]
    global_entropy_by_dist: Dict[str, float]
    global_u_sum_by_dist: Dict[str, float]
    global_delta_max_by_dist: Dict[str, float]
    tail_sensitivity_by_dist: Dict[str, float]
    tail_std_r_fusion_by_dist: Dict[str, float]
