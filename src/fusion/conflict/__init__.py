"""
src/fusion/conflict/__init__.py
Phase C11.7: Cross-Modality Conflict & Discordance Analysis

Public API exports for conflict analysis.
"""

from .conflict_result import PairwiseRecord, ConflictResult
from .pairwise_disagreement import compute_pairwise_record, compute_all_pairwise_records
from .conflict_metrics import (
    compute_max_disagreement,
    compute_mean_disagreement,
    compute_weighted_dispersion,
    compute_weight_entropy,
)
from .conflict_classifier import (
    classify_conflict_severity,
    OPERATIONAL_LOW_THRESHOLD,
    OPERATIONAL_HIGH_THRESHOLD,
)
from .conflict_explainability import generate_conflict_summary
from .conflict_engine import ConflictEngine

__all__ = [
    "PairwiseRecord",
    "ConflictResult",
    "compute_pairwise_record",
    "compute_all_pairwise_records",
    "compute_max_disagreement",
    "compute_mean_disagreement",
    "compute_weighted_dispersion",
    "compute_weight_entropy",
    "classify_conflict_severity",
    "OPERATIONAL_LOW_THRESHOLD",
    "OPERATIONAL_HIGH_THRESHOLD",
    "generate_conflict_summary",
    "ConflictEngine",
]
