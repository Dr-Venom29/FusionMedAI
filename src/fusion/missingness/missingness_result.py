"""
src/fusion/missingness/missingness_result.py
Phase C11.8: Missingness Result Schema & Contracts

Defines immutable, self-validating data structures for individual regime evaluations,
authority redistribution records, stress scenarios, and comparative baseline summaries.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import math


@dataclass(frozen=True)
class RegimeEvaluation:
    """
    Immutable evaluation of a single decision packet under a specific availability regime.
    """
    packet_id: str
    regime_name: str
    active_modalities: Tuple[str, ...]
    num_active: int
    weights: Dict[str, float]
    r_fusion: float
    dcri: float
    delta: float
    u_sum: float
    u_mean: float
    entropy: float
    w_max: float
    dominant_modality: str
    status: str = "SUCCESS"

    def __post_init__(self) -> None:
        if self.num_active == 0:
            if self.status != "NO_MODALITY_AVAILABLE":
                raise ValueError("Zero-modality evaluation must have status='NO_MODALITY_AVAILABLE'")
            if self.r_fusion != 0.0 or self.dcri != 0.0 or self.u_sum != 0.0:
                raise ValueError("Zero-modality evaluation must have zero numerical outputs")
        else:
            if self.status != "SUCCESS":
                raise ValueError(f"Active evaluation must have status='SUCCESS', got {self.status}")
            if not (0.0 <= self.r_fusion <= 1.0 + 1e-6):
                raise ValueError(f"r_fusion={self.r_fusion} out of bounds [0, 1]")
            # Check DCRI theoretical bound: [-delta * M, 1.0]
            lower_bound = -self.delta * self.num_active - 1e-6
            if not (lower_bound <= self.dcri <= 1.0 + 1e-6):
                raise ValueError(f"dcri={self.dcri} out of bounds [{lower_bound}, 1.0]")
            # Check simplex sum
            active_w_sum = sum(self.weights.get(m, 0.0) for m in self.active_modalities)
            if abs(active_w_sum - 1.0) > 1e-5:
                raise ValueError(f"Active weights sum to {active_w_sum}, expected 1.0")


@dataclass(frozen=True)
class RedistributionRecord:
    """
    Comparative record contrasting full tri-modal availability against a subset regime.
    """
    packet_id: str
    subset_regime: str
    removed_modalities: Tuple[str, ...]
    remaining_modalities: Tuple[str, ...]
    delta_w: Dict[str, float]
    delta_r_signed: float
    delta_r_abs: float
    delta_u_sum: float
    delta_penalty: float
    delta_dcri_signed: float
    delta_dcri_abs: float
    entropy_shift: float
    w_max_shift: float


@dataclass(frozen=True)
class StressScenarioRecord:
    """
    Evaluation record for targeted stress missingness dropouts.
    """
    scenario_name: str
    criterion: str
    dropped_modality: str
    remaining_modalities: Tuple[str, ...]
    r_fusion_full: float
    r_fusion_stress: float
    delta_r: float
    dcri_full: float
    dcri_stress: float
    delta_dcri: float
    weights_full: Dict[str, float]
    weights_stress: Dict[str, float]


@dataclass(frozen=True)
class MaskedInvarianceRecord:
    """
    Record verifying that mutating unavailable modality scalars has zero effect on active outputs.
    """
    packet_id: str
    unavailable_modality: str
    perturbation_type: str
    active_weights_match: bool
    r_fusion_match: bool
    dcri_match: bool
    max_weight_diff: float
    r_fusion_diff: float
    dcri_diff: float
    passed: bool
