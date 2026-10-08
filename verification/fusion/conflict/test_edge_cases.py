"""
verification/fusion/conflict/test_edge_cases.py
Unit tests for boundary conditions, extreme risks, zero variance, and contract validation.
"""

import pytest
import math
from src.fusion.baselines.decision_packet import ControlledDecisionPacket, ModalityRecord
from src.fusion.conflict.conflict_engine import ConflictEngine
from src.fusion.conflict.conflict_result import PairwiseRecord, ConflictResult


def test_extreme_polar_conflict():
    pkt = ControlledDecisionPacket(
        packet_id="polar_packet",
        retina=ModalityRecord(
            sample_id="retina_polar",
            modality="retina",
            risk=1.00,
            calibrated_probability=1.00,
            confidence=1.00,
            uncertainty=0.00,
            quality=1.00,
            availability=1,
            reliability=0.929956,
            model_version="retina_v1",
        ),
        foot=ModalityRecord(
            sample_id="foot_polar",
            modality="foot",
            risk=0.00,
            calibrated_probability=0.00,
            confidence=1.00,
            uncertainty=0.00,
            quality=1.00,
            availability=1,
            reliability=0.922266,
            model_version="foot_v1",
        ),
        clinical=ModalityRecord(
            sample_id="clinical_polar",
            modality="clinical",
            risk=0.00,
            calibrated_probability=0.00,
            confidence=0.00,
            uncertainty=0.00,
            quality=0.00,
            availability=0,
            reliability=0.825382,
            model_version="clinical_v1",
        ),
        seed=115,
    )

    engine = ConflictEngine()
    res = engine.evaluate_packet(pkt, delta=0.20)

    assert pytest.approx(res.max_disagreement, abs=1e-6) == 1.00
    assert pytest.approx(res.mean_disagreement, abs=1e-6) == 1.00
    assert res.conflict_severity == "HIGH"
    assert res.dominant_conflict_pair == ("retina", "foot")


def test_contract_nan_rejection():
    with pytest.raises(ValueError, match="must be finite"):
        PairwiseRecord(
            modality_a="retina",
            modality_b="foot",
            risk_a=float("nan"),
            risk_b=0.50,
            signed_difference=0.0,
            absolute_difference=0.0,
            higher_risk_modality="tied",
            lower_risk_modality="tied",
            weight_a=0.50,
            weight_b=0.50,
            dominant_authority_modality="tied",
            uncertainty_a=0.10,
            uncertainty_b=0.10,
            reliability_a=0.90,
            reliability_b=0.90,
            reliability_authority_aligned=True,
        )
