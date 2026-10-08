"""
src/fusion/combination_analysis/head_tail_classifier.py
Phase C11.9: Head, Middle, and Tail Classification Rules

Assigns combinations into structural frequency tiers:
1. HEAD: High-frequency combinations (Rank 1–2; e.g. RFC, RF).
2. MIDDLE: Moderate-frequency combinations (Rank 3–4; e.g. RC, FC).
3. TAIL: Low-frequency / rare combinations (Rank 5–7; e.g. R, F, C).

Provides both deterministic rank-based categorization and threshold-based verification.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

from src.fusion.combination_analysis.combination_definition import NON_EMPTY_COMBINATIONS
from src.fusion.combination_analysis.distribution_generator import (
    DistributionConfig,
    DISTRIBUTION_D1_BALANCED,
    DISTRIBUTION_D2_MODERATE_HEAD_TAIL,
    DISTRIBUTION_D3_STRONG_LONG_TAIL,
    get_distribution_config,
)

TIER_HEAD = "HEAD"
TIER_MIDDLE = "MIDDLE"
TIER_TAIL = "TAIL"
TIER_UNIFORM = "UNIFORM"

ALL_TIERS = (TIER_HEAD, TIER_MIDDLE, TIER_TAIL, TIER_UNIFORM)


@dataclass(frozen=True)
class CombinationTierRecord:
    """Immutable classification record for a single combination in a distribution."""
    combination_id: str
    distribution_id: str
    probability: float
    rank: int                     # 1-indexed rank by probability (1 = highest)
    tier: str                     # 'HEAD', 'MIDDLE', 'TAIL', or 'UNIFORM'
    threshold_tier: str           # Tier assigned by threshold rule


@dataclass(frozen=True)
class DistributionTierAssignment:
    """Complete tier assignment for a distribution configuration."""
    distribution_id: str
    head_combinations: Tuple[str, ...]
    middle_combinations: Tuple[str, ...]
    tail_combinations: Tuple[str, ...]
    head_to_tail_ratio: Optional[float]
    records: Dict[str, CombinationTierRecord]


def classify_distribution_tiers(dist_config: DistributionConfig) -> DistributionTierAssignment:
    """
    Deterministically categorizes the 7 non-empty combinations into HEAD, MIDDLE, and TAIL
    based on the pre-registered rank and frequency rules.
    """
    probs = dist_config.probabilities

    # Sort combinations by descending probability, using canonical order as stable tie-breaker
    canonical_indices = {c: i for i, c in enumerate(NON_EMPTY_COMBINATIONS)}
    sorted_combs = sorted(
        NON_EMPTY_COMBINATIONS,
        key=lambda c: (-probs[c], canonical_indices[c])
    )

    records: Dict[str, CombinationTierRecord] = {}
    head_list: List[str] = []
    middle_list: List[str] = []
    tail_list: List[str] = []

    is_balanced = (dist_config.distribution_id == DISTRIBUTION_D1_BALANCED)

    for rank_0, c in enumerate(sorted_combs):
        rank = rank_0 + 1
        p = probs[c]

        # Threshold-based rule:
        # Head >= 0.20, Middle [0.05, 0.20), Tail < 0.05
        if p >= 0.20:
            thresh_tier = TIER_HEAD
        elif p >= 0.05:
            thresh_tier = TIER_MIDDLE
        else:
            thresh_tier = TIER_TAIL

        # Rank-based rule:
        if is_balanced:
            tier = TIER_UNIFORM
            head_list.append(c)
        else:
            if rank <= 2:
                tier = TIER_HEAD
                head_list.append(c)
            elif rank <= 4:
                tier = TIER_MIDDLE
                middle_list.append(c)
            else:
                tier = TIER_TAIL
                tail_list.append(c)

        records[c] = CombinationTierRecord(
            combination_id=c,
            distribution_id=dist_config.distribution_id,
            probability=p,
            rank=rank,
            tier=tier,
            threshold_tier=thresh_tier,
        )

    # Compute Head-to-Tail Ratio (HTR)
    if is_balanced or not tail_list:
        htr = None
    else:
        p_head = sum(probs[c] for c in head_list)
        p_tail = sum(probs[c] for c in tail_list)
        htr = round(float(p_head / p_tail), 6) if p_tail > 0 else float("inf")

    return DistributionTierAssignment(
        distribution_id=dist_config.distribution_id,
        head_combinations=tuple(head_list),
        middle_combinations=tuple(middle_list),
        tail_combinations=tuple(tail_list),
        head_to_tail_ratio=htr,
        records=records,
    )
