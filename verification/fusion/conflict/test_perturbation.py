"""
verification/fusion/conflict/test_perturbation.py
Unit tests for controlled risk perturbations and conflict response monotonicity.
"""

import pytest
from src.fusion.baselines.decision_packet import ControlledDecisionPacket, ModalityRecord
from src.fusion.conflict.conflict_engine import ConflictEngine


@pytest.fixture
def base_packet() -> ControlledDecisionPacket:
    return ControlledDecisionPacket(
        packet_id="perturb_packet_001",
        retina=ModalityRecord(
            sample_id="retina_p001",
            modality="retina",
            risk=0.50,
            calibrated_probability=0.50,
            confidence=0.80,
            uncertainty=0.10,
            quality=0.90,
            availability=1,
            reliability=0.929956,
            model_version="retina_v1",
        ),
        foot=ModalityRecord(
            sample_id="foot_p001",
            modality="foot",
            risk=0.50,
            calibrated_probability=0.50,
            confidence=0.80,
            uncertainty=0.10,
            quality=0.90,
            availability=1,
            reliability=0.922266,
            model_version="foot_v1",
        ),
        clinical=ModalityRecord(
            sample_id="clinical_p001",
            modality="clinical",
            risk=0.50,
            calibrated_probability=0.50,
            confidence=0.80,
            uncertainty=0.10,
            quality=0.90,
            availability=1,
            reliability=0.825382,
            model_version="clinical_v1",
        ),
        seed=115,
    )


def test_perturbation_outward_monotonicity(base_packet):
    """
    Starting from complete consensus (r=0.50, 0.50, 0.50),
    perturbing clinical risk outward (0.50 -> 0.60 -> 0.80 -> 1.00)
    must strictly increase max_disagreement and weighted_std.
    """
    engine = ConflictEngine()
    
    res_050 = engine.perturb_modality_risk(base_packet, target_modality="clinical", new_risk=0.50)
    res_060 = engine.perturb_modality_risk(base_packet, target_modality="clinical", new_risk=0.60)
    res_080 = engine.perturb_modality_risk(base_packet, target_modality="clinical", new_risk=0.80)
    res_100 = engine.perturb_modality_risk(base_packet, target_modality="clinical", new_risk=1.00)

    assert res_050.max_disagreement == 0.0
    assert pytest.approx(res_060.max_disagreement, abs=1e-5) == 0.10
    assert pytest.approx(res_080.max_disagreement, abs=1e-5) == 0.30
    assert pytest.approx(res_100.max_disagreement, abs=1e-5) == 0.50

    assert res_050.weighted_std < res_060.weighted_std < res_080.weighted_std < res_100.weighted_std


def test_perturbation_inward_monotonicity():
    """
    Starting from high conflict (r=0.90, 0.10, 0.10),
    moving retina inward (0.90 -> 0.70 -> 0.50 -> 0.10)
    must strictly decrease max_disagreement.
    """
    engine = ConflictEngine()
    pkt = ControlledDecisionPacket(
        packet_id="inward_packet",
        retina=ModalityRecord(
            sample_id="retina_inward",
            modality="retina",
            risk=0.90,
            calibrated_probability=0.90,
            confidence=0.80,
            uncertainty=0.10,
            quality=0.90,
            availability=1,
            reliability=0.929956,
            model_version="retina_v1",
        ),
        foot=ModalityRecord(
            sample_id="foot_inward",
            modality="foot",
            risk=0.10,
            calibrated_probability=0.10,
            confidence=0.80,
            uncertainty=0.10,
            quality=0.90,
            availability=1,
            reliability=0.922266,
            model_version="foot_v1",
        ),
        clinical=ModalityRecord(
            sample_id="clinical_inward",
            modality="clinical",
            risk=0.10,
            calibrated_probability=0.10,
            confidence=0.80,
            uncertainty=0.10,
            quality=0.90,
            availability=1,
            reliability=0.825382,
            model_version="clinical_v1",
        ),
        seed=115,
    )

    res_090 = engine.perturb_modality_risk(pkt, target_modality="retina", new_risk=0.90)
    res_070 = engine.perturb_modality_risk(pkt, target_modality="retina", new_risk=0.70)
    res_050 = engine.perturb_modality_risk(pkt, target_modality="retina", new_risk=0.50)
    res_010 = engine.perturb_modality_risk(pkt, target_modality="retina", new_risk=0.10)

    assert res_090.max_disagreement > res_070.max_disagreement > res_050.max_disagreement > res_010.max_disagreement
    assert res_010.max_disagreement == 0.0
