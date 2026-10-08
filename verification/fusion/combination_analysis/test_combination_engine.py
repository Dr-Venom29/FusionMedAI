"""
verification/fusion/combination_analysis/test_combination_engine.py
Phase C11.9: Integration tests for ModalityCombinationEngine execution.
"""

import pytest
import numpy as np

from src.fusion.combination_analysis.combination_engine import ModalityCombinationEngine
from src.fusion.combination_analysis.distribution_generator import (
    ALL_DISTRIBUTIONS,
    DISTRIBUTION_D2_MODERATE_HEAD_TAIL,
)
from src.fusion.reliability.global_reliability import (
    FROZEN_RETINA_RELIABILITY,
    FROZEN_FOOT_RELIABILITY,
    FROZEN_CLINICAL_RELIABILITY,
)
from src.fusion.baselines.decision_packet import ControlledDecisionPacket, ModalityRecord


def make_mini_cohort(n: int = 14) -> list[ControlledDecisionPacket]:
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
        cohort.append(ControlledDecisionPacket(packet_id=f"pkt_{i:02d}", retina=r, foot=f, clinical=c, seed=115))
    return cohort


def test_combination_engine_evaluate_distribution():
    cohort = make_mini_cohort(14)
    engine = ModalityCombinationEngine(delta=0.20, bootstrap_resamples=50, seed=115)

    rec = engine.evaluate_distribution(cohort, DISTRIBUTION_D2_MODERATE_HEAD_TAIL)

    assert rec.distribution_id == DISTRIBUTION_D2_MODERATE_HEAD_TAIL
    assert rec.total_packets == 14
    assert len(rec.combinations) == 7
    assert "RFC" in rec.combinations
    assert rec.global_mean_r_fusion >= 0.0


def test_combination_engine_evaluate_all_distributions():
    cohort = make_mini_cohort(14)
    engine = ModalityCombinationEngine(delta=0.20, bootstrap_resamples=50, seed=115)

    dist_recs, cross_sens = engine.evaluate_all_distributions(cohort)

    assert set(dist_recs.keys()) == set(ALL_DISTRIBUTIONS)
    assert len(cross_sens.distributions_evaluated) == 3
