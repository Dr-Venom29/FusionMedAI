"""
verification/fusion/combination_analysis/test_head_tail_classifier.py
Phase C11.9: Unit tests for deterministic Head, Middle, and Tail tier classification.
"""

import pytest
from src.fusion.combination_analysis.distribution_generator import (
    DISTRIBUTION_D1_BALANCED,
    DISTRIBUTION_D2_MODERATE_HEAD_TAIL,
    DISTRIBUTION_D3_STRONG_LONG_TAIL,
    get_distribution_config,
)
from src.fusion.combination_analysis.head_tail_classifier import (
    classify_distribution_tiers,
    TIER_HEAD,
    TIER_MIDDLE,
    TIER_TAIL,
    TIER_UNIFORM,
)


def test_d1_balanced_classification():
    cfg = get_distribution_config(DISTRIBUTION_D1_BALANCED)
    tiers = classify_distribution_tiers(cfg)

    # In balanced distribution, all 7 combinations are classified uniformly and HTR is None
    assert tiers.head_to_tail_ratio is None
    for c, r in tiers.records.items():
        assert r.tier == TIER_UNIFORM


def test_d2_moderate_classification():
    cfg = get_distribution_config(DISTRIBUTION_D2_MODERATE_HEAD_TAIL)
    tiers = classify_distribution_tiers(cfg)

    assert tiers.head_combinations == ("RFC", "RF")
    assert tiers.middle_combinations == ("RC", "FC")
    assert tiers.tail_combinations == ("R", "F", "C")
    assert tiers.head_to_tail_ratio == 4.0  # 60% / 15% = 4.0


def test_d3_strong_tail_classification():
    cfg = get_distribution_config(DISTRIBUTION_D3_STRONG_LONG_TAIL)
    tiers = classify_distribution_tiers(cfg)

    assert tiers.head_combinations == ("RFC", "RF")
    assert tiers.middle_combinations == ("RC", "FC")
    assert tiers.tail_combinations == ("R", "F", "C")
    # Head: 75%, Tail: 7% => 75 / 7 ~ 10.714286
    assert tiers.head_to_tail_ratio > 10.0


def test_threshold_tier_conformance():
    cfg = get_distribution_config(DISTRIBUTION_D2_MODERATE_HEAD_TAIL)
    tiers = classify_distribution_tiers(cfg)

    # Threshold rule: >= 0.20 Head, [0.05, 0.20) Middle, < 0.05 Tail
    assert tiers.records["RFC"].threshold_tier == TIER_HEAD      # 0.35
    assert tiers.records["RF"].threshold_tier == TIER_HEAD       # 0.25
    assert tiers.records["RC"].threshold_tier == TIER_MIDDLE     # 0.15
    assert tiers.records["FC"].threshold_tier == TIER_MIDDLE     # 0.10
    assert tiers.records["R"].threshold_tier == TIER_MIDDLE      # 0.06
    assert tiers.records["F"].threshold_tier == TIER_MIDDLE      # 0.05
    assert tiers.records["C"].threshold_tier == TIER_TAIL        # 0.04
