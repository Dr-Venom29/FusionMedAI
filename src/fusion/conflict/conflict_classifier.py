"""
src/fusion/conflict/conflict_classifier.py
Phase C11.7: Cross-Modality Conflict & Discordance Analysis

Implements pre-specified operational conflict severity classification.
Thresholds represent operational conflict-alert guidelines, not validated clinical boundaries:
- LOW: Delta_max < 0.20
- MODERATE: 0.20 <= Delta_max < 0.35
- HIGH: Delta_max >= 0.35
"""

from typing import Optional

# Pre-specified operational thresholds
OPERATIONAL_LOW_THRESHOLD = 0.20
OPERATIONAL_HIGH_THRESHOLD = 0.35


def classify_conflict_severity(
    num_active: int,
    max_disagreement: Optional[float],
) -> str:
    """
    Classifies conflict severity into pre-specified operational bands:
    - "NO_MODALITY_AVAILABLE": num_active == 0
    - "NOT_APPLICABLE": num_active == 1 (pairwise discordance cannot be computed)
    - "LOW": num_active >= 2 and Delta_max < 0.20
    - "MODERATE": num_active >= 2 and 0.20 <= Delta_max < 0.35
    - "HIGH": num_active >= 2 and Delta_max >= 0.35
    """
    if num_active == 0:
        return "NO_MODALITY_AVAILABLE"

    if num_active == 1:
        return "NOT_APPLICABLE"

    if max_disagreement is None:
        raise ValueError("max_disagreement cannot be None when num_active >= 2")

    if max_disagreement < OPERATIONAL_LOW_THRESHOLD:
        return "LOW"
    elif max_disagreement < OPERATIONAL_HIGH_THRESHOLD:
        return "MODERATE"
    else:
        return "HIGH"
