"""
src/fusion/calibration/calibration_result.py
Phase C11.11: Structured Results Contracts for Fusion Calibration Benchmark
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional


@dataclass(frozen=True)
class ModalityCalibrationProfile:
    """Modality-level calibration evaluation profile based on true validation ground truth."""
    modality: str
    calibration_method: str
    raw_ece: float
    calibrated_ece: float
    raw_brier: float
    calibrated_brier: float
    raw_nll: float
    calibrated_nll: float
    calibration_slope: float
    calibration_intercept: float
    raw_mean_confidence: float
    calibrated_mean_confidence: float
    raw_mean_entropy: float
    calibrated_mean_entropy: float
    ece_reduction_percent: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality,
            "calibration_method": self.calibration_method,
            "raw_ece": round(float(self.raw_ece), 6),
            "calibrated_ece": round(float(self.calibrated_ece), 6),
            "raw_brier": round(float(self.raw_brier), 6),
            "calibrated_brier": round(float(self.calibrated_brier), 6),
            "raw_nll": round(float(self.raw_nll), 6),
            "calibrated_nll": round(float(self.calibrated_nll), 6),
            "calibration_slope": round(float(self.calibration_slope), 6),
            "calibration_intercept": round(float(self.calibration_intercept), 6),
            "raw_mean_confidence": round(float(self.raw_mean_confidence), 6),
            "calibrated_mean_confidence": round(float(self.calibrated_mean_confidence), 6),
            "raw_mean_entropy": round(float(self.raw_mean_entropy), 6),
            "calibrated_mean_entropy": round(float(self.calibrated_mean_entropy), 6),
            "ece_reduction_percent": round(float(self.ece_reduction_percent), 2),
        }


@dataclass(frozen=True)
class PacketCalibrationResult:
    """Individual decision packet evaluation comparing uncalibrated vs calibrated fusion."""
    packet_id: str
    condition: str
    weights: Dict[str, float]
    r_fusion: float
    dcri: float
    routing_entropy: float
    conflict_index: float
    dominant_modality: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "packet_id": self.packet_id,
            "condition": self.condition,
            "weights": {k: round(float(v), 6) for k, v in self.weights.items()},
            "r_fusion": round(float(self.r_fusion), 6),
            "dcri": round(float(self.dcri), 6),
            "routing_entropy": round(float(self.routing_entropy), 6),
            "conflict_index": round(float(self.conflict_index), 6),
            "dominant_modality": self.dominant_modality,
        }


@dataclass(frozen=True)
class CohortCalibrationSummary:
    """Cohort-level summary of decision metrics under a specific calibration condition."""
    condition: str
    is_calibrated: bool
    mean_weights: Dict[str, float]
    mean_r_fusion: float
    std_r_fusion: float
    mean_dcri: float
    std_dcri: float
    mean_routing_entropy: float
    mean_conflict_index: float
    dominant_modality_distribution: Dict[str, float]
    sample_size: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition": self.condition,
            "is_calibrated": self.is_calibrated,
            "mean_weights": {k: round(float(v), 6) for k, v in self.mean_weights.items()},
            "mean_r_fusion": round(float(self.mean_r_fusion), 6),
            "std_r_fusion": round(float(self.std_r_fusion), 6),
            "mean_dcri": round(float(self.mean_dcri), 6),
            "std_dcri": round(float(self.std_dcri), 6),
            "mean_routing_entropy": round(float(self.mean_routing_entropy), 6),
            "mean_conflict_index": round(float(self.mean_conflict_index), 6),
            "dominant_modality_distribution": {
                k: round(float(v), 4) for k, v in self.dominant_modality_distribution.items()
            },
            "sample_size": self.sample_size,
        }


@dataclass(frozen=True)
class PairedCalibrationDelta:
    """Statistical summary of paired differences (Calibrated - Uncalibrated)."""
    metric_name: str
    comparison_name: str
    mean_delta: float
    median_delta: float
    std_delta: float
    bootstrap_ci_lower: float
    bootstrap_ci_upper: float
    excludes_zero: bool
    hypothesis_id: str
    interpretation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "comparison_name": self.comparison_name,
            "mean_delta": round(float(self.mean_delta), 6),
            "median_delta": round(float(self.median_delta), 6),
            "std_delta": round(float(self.std_delta), 6),
            "bootstrap_ci_lower": round(float(self.bootstrap_ci_lower), 6),
            "bootstrap_ci_upper": round(float(self.bootstrap_ci_upper), 6),
            "excludes_zero": self.excludes_zero,
            "hypothesis_id": self.hypothesis_id,
            "interpretation": self.interpretation,
        }
