"""
verification/fusion/combination_analysis/test_combination_edge_cases.py
Phase C11.9: Unit tests for edge cases, zero-modality fail-closed handling, and numerical boundaries.
"""

import pytest
import numpy as np

from src.fusion.combination_analysis.combination_definition import (
    COMBINATION_EMPTY,
    apply_combination_to_packet,
)
from src.fusion.reliability.global_reliability import (
    FROZEN_RETINA_RELIABILITY,
    FROZEN_FOOT_RELIABILITY,
    FROZEN_CLINICAL_RELIABILITY,
)
from src.fusion.dcri.dcri_engine import DCRIEngine
from src.fusion.baselines.decision_packet import ControlledDecisionPacket, ModalityRecord


def test_empty_combination_fails_closed():
    r = ModalityRecord(
        sample_id="s1", modality="retina", risk=0.3, calibrated_probability=(0.7, 0.3),
        confidence=0.8, uncertainty=0.1, quality=0.9, availability=True,
        reliability=FROZEN_RETINA_RELIABILITY, model_version="v1"
    )
    f = ModalityRecord(
        sample_id="s1", modality="foot", risk=0.5, calibrated_probability=(0.5, 0.5),
        confidence=0.7, uncertainty=0.2, quality=0.8, availability=True,
        reliability=FROZEN_FOOT_RELIABILITY, model_version="v1"
    )
    c = ModalityRecord(
        sample_id="s1", modality="clinical", risk=0.2, calibrated_probability=(0.8, 0.2),
        confidence=0.9, uncertainty=0.05, quality=0.95, availability=True,
        reliability=FROZEN_CLINICAL_RELIABILITY, model_version="v1"
    )
    pkt = ControlledDecisionPacket(packet_id="dummy_01", retina=r, foot=f, clinical=c, seed=115)

    empty_pkt = apply_combination_to_packet(pkt, COMBINATION_EMPTY)
    engine = DCRIEngine()
    res = engine.evaluate_packet(empty_pkt, delta=0.20)

    assert res.status == "NO_MODALITY_AVAILABLE"
    assert res.r_fusion == 0.0
    assert res.dcri == 0.0
    assert res.u_sum == 0.0
