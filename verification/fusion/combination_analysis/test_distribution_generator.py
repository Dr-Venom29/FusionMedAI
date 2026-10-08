"""
verification/fusion/combination_analysis/test_distribution_generator.py
Phase C11.9: Unit tests for controlled frequency distribution configurations and stratified cohort generation.
"""

import pytest
import numpy as np

from src.fusion.combination_analysis.distribution_generator import (
    ALL_DISTRIBUTIONS,
    DISTRIBUTION_D1_BALANCED,
    DISTRIBUTION_D2_MODERATE_HEAD_TAIL,
    DISTRIBUTION_D3_STRONG_LONG_TAIL,
    get_distribution_config,
    generate_stratified_distribution_cohort,
)
from src.fusion.reliability.global_reliability import (
    FROZEN_RETINA_RELIABILITY,
    FROZEN_FOOT_RELIABILITY,
    FROZEN_CLINICAL_RELIABILITY,
)
from src.fusion.combination_analysis.combination_definition import NON_EMPTY_COMBINATIONS
from src.fusion.baselines.decision_packet import ControlledDecisionPacket, ModalityRecord


def make_dummy_cohort(n: int = 500) -> list[ControlledDecisionPacket]:
    cohort = []
    for i in range(n):
        r = ModalityRecord(
            sample_id=f"s_{i}", modality="retina", risk=0.25, calibrated_probability=(0.75, 0.25),
            confidence=0.8, uncertainty=0.1, quality=0.9, availability=True,
            reliability=FROZEN_RETINA_RELIABILITY, model_version="v1"
        )
        f = ModalityRecord(
            sample_id=f"s_{i}", modality="foot", risk=0.50, calibrated_probability=(0.50, 0.50),
            confidence=0.7, uncertainty=0.2, quality=0.8, availability=True,
            reliability=FROZEN_FOOT_RELIABILITY, model_version="v1"
        )
        c = ModalityRecord(
            sample_id=f"s_{i}", modality="clinical", risk=0.10, calibrated_probability=(0.90, 0.10),
            confidence=0.9, uncertainty=0.05, quality=0.95, availability=True,
            reliability=FROZEN_CLINICAL_RELIABILITY, model_version="v1"
        )
        cohort.append(ControlledDecisionPacket(packet_id=f"pkt_{i:04d}", retina=r, foot=f, clinical=c, seed=115))
    return cohort


def test_distribution_configs_sum_to_one():
    for dist_id in ALL_DISTRIBUTIONS:
        cfg = get_distribution_config(dist_id)
        total_p = sum(cfg.probabilities.values())
        assert np.isclose(total_p, 1.0, atol=1e-6)
        assert len(cfg.probabilities) == 7


def test_expected_counts_n500_sum_to_500():
    for dist_id in ALL_DISTRIBUTIONS:
        cfg = get_distribution_config(dist_id)
        total_count = sum(cfg.expected_counts_n500.values())
        assert total_count == 500


def test_stratified_generation_deterministic():
    cohort = make_dummy_cohort(500)
    assigned_1 = generate_stratified_distribution_cohort(cohort, DISTRIBUTION_D2_MODERATE_HEAD_TAIL, seed=115)
    assigned_2 = generate_stratified_distribution_cohort(cohort, DISTRIBUTION_D2_MODERATE_HEAD_TAIL, seed=115)

    assert len(assigned_1) == 500
    for a1, a2 in zip(assigned_1, assigned_2):
        assert a1.original_packet_id == a2.original_packet_id
        assert a1.assigned_combination == a2.assigned_combination


def test_stratified_generation_counts():
    cohort = make_dummy_cohort(500)
    for dist_id in ALL_DISTRIBUTIONS:
        cfg = get_distribution_config(dist_id)
        assigned = generate_stratified_distribution_cohort(cohort, dist_id, seed=115)

        counts = {}
        for ap in assigned:
            c = ap.assigned_combination
            counts[c] = counts.get(c, 0) + 1

        assert counts == cfg.expected_counts_n500
