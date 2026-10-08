"""
src/fusion/degradation/degradation_result.py
Phase C11.10: Data Structures for Input Degradation Results & Metrics

Defines immutable structured dataclasses for:
- Quality response records (Experiment A)
- Routing authority response records (Experiment B)
- Quality-Authority response slope records (S_QW)
- Quality-Routing alignment records (rho_QW)
- Monotonicity test records
- Baseline comparison records (B1–B6, B5 vs B6)
- Comprehensive degradation experiment result container
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple


@dataclass(frozen=True)
class QualityResponseRecord:
    """Stores statistical summary of quality metric response under degradation."""
    modality: str
    operator: str
    severity: str
    n_packets: int
    mean_q_clean: float
    mean_q_degraded: float
    mean_delta_q: float
    std_delta_q: float
    median_delta_q: float
    ci_95_delta_q: Tuple[float, float]
    quality_loss_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality,
            "operator": self.operator,
            "severity": self.severity,
            "n_packets": self.n_packets,
            "mean_q_clean": round(self.mean_q_clean, 6),
            "mean_q_degraded": round(self.mean_q_degraded, 6),
            "mean_delta_q": round(self.mean_delta_q, 6),
            "std_delta_q": round(self.std_delta_q, 6),
            "median_delta_q": round(self.median_delta_q, 6),
            "ci_95_delta_q": [round(self.ci_95_delta_q[0], 6), round(self.ci_95_delta_q[1], 6)],
            "quality_loss_pct": round(self.quality_loss_pct, 2),
        }


@dataclass(frozen=True)
class RoutingResponseRecord:
    """Stores statistical summary of ACARA-U routing authority response under degradation."""
    modality: str
    operator: str
    severity: str
    n_packets: int
    mean_w_clean: float
    mean_w_degraded: float
    mean_delta_w: float
    std_delta_w: float
    median_delta_w: float
    ci_95_delta_w: Tuple[float, float]
    mean_rar: float
    mean_entropy_clean: float
    mean_entropy_degraded: float
    mean_delta_entropy: float
    redistributed_authority_mean: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality,
            "operator": self.operator,
            "severity": self.severity,
            "n_packets": self.n_packets,
            "mean_w_clean": round(self.mean_w_clean, 6),
            "mean_w_degraded": round(self.mean_w_degraded, 6),
            "mean_delta_w": round(self.mean_delta_w, 6),
            "std_delta_w": round(self.std_delta_w, 6),
            "median_delta_w": round(self.median_delta_w, 6),
            "ci_95_delta_w": [round(self.ci_95_delta_w[0], 6), round(self.ci_95_delta_w[1], 6)],
            "mean_rar": round(self.mean_rar, 6),
            "mean_entropy_clean": round(self.mean_entropy_clean, 6),
            "mean_entropy_degraded": round(self.mean_entropy_degraded, 6),
            "mean_delta_entropy": round(self.mean_delta_entropy, 6),
            "redistributed_authority_mean": round(self.redistributed_authority_mean, 6),
        }


@dataclass(frozen=True)
class SlopeRecord:
    """Stores Quality-Authority Response Slope S_QW = Delta w_i / Delta Q_i."""
    modality: str
    operator: str
    severity: str
    mean_slope: float
    median_slope: float
    ci_95_slope: Tuple[float, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality,
            "operator": self.operator,
            "severity": self.severity,
            "mean_slope": round(self.mean_slope, 6),
            "median_slope": round(self.median_slope, 6),
            "ci_95_slope": [round(self.ci_95_slope[0], 6), round(self.ci_95_slope[1], 6)],
        }


@dataclass(frozen=True)
class MonotonicityRecord:
    """Stores packet-level monotonicity rate for quality and routing response."""
    modality: str
    operator: str
    total_eligible_packets: int
    quality_monotonic_packets: int
    quality_monotonic_rate: float
    routing_monotonic_packets: int
    routing_monotonic_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality,
            "operator": self.operator,
            "total_eligible_packets": self.total_eligible_packets,
            "quality_monotonic_packets": self.quality_monotonic_packets,
            "quality_monotonic_rate": round(self.quality_monotonic_rate, 4),
            "routing_monotonic_packets": self.routing_monotonic_packets,
            "routing_monotonic_rate": round(self.routing_monotonic_rate, 4),
        }


@dataclass(frozen=True)
class BaselineComparisonRecord:
    """Stores comparative metrics across Baselines B1–B6 at a specific severity level."""
    modality: str
    operator: str
    severity: str
    baseline_id: str
    baseline_name: str
    mean_w_degraded: float
    mean_delta_w: float
    mean_delta_risk: float
    b6_vs_b5_delta_w: float
    b6_vs_b5_paired_ci: Tuple[float, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality,
            "operator": self.operator,
            "severity": self.severity,
            "baseline_id": self.baseline_id,
            "baseline_name": self.baseline_name,
            "mean_w_degraded": round(self.mean_w_degraded, 6),
            "mean_delta_w": round(self.mean_delta_w, 6),
            "mean_delta_risk": round(self.mean_delta_risk, 6),
            "b6_vs_b5_delta_w": round(self.b6_vs_b5_delta_w, 6),
            "b6_vs_b5_paired_ci": [round(self.b6_vs_b5_paired_ci[0], 6), round(self.b6_vs_b5_paired_ci[1], 6)],
        }
