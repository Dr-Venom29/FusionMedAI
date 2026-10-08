"""
verification/fusion/combination_analysis/test_combination_definition.py
Phase C11.9: Unit tests for modality combination definitions and immutability contracts.
"""

import pytest
from src.fusion.combination_analysis.combination_definition import (
    ALL_COMBINATIONS,
    NON_EMPTY_COMBINATIONS,
    BIMODAL_COMBINATIONS,
    UNIMODAL_COMBINATIONS,
    COMBINATION_RFC,
    COMBINATION_RF,
    COMBINATION_RC,
    COMBINATION_FC,
    COMBINATION_R,
    COMBINATION_F,
    COMBINATION_C,
    COMBINATION_EMPTY,
    COMBINATION_MASKS,
    COMBINATION_CARDINALITY,
    COMBINATION_ACTIVE_MODALITIES,
    get_combination_spec,
    apply_combination_to_packet,
)
from src.fusion.reliability.global_reliability import (
    FROZEN_RETINA_RELIABILITY,
    FROZEN_FOOT_RELIABILITY,
    FROZEN_CLINICAL_RELIABILITY,
)
from src.fusion.baselines.decision_packet import ControlledDecisionPacket, ModalityRecord


def make_dummy_packet() -> ControlledDecisionPacket:
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
    return ControlledDecisionPacket(packet_id="dummy_01", retina=r, foot=f, clinical=c, seed=115)


def test_combination_count_and_cardinality():
    assert len(ALL_COMBINATIONS) == 8
    assert len(NON_EMPTY_COMBINATIONS) == 7
    assert len(BIMODAL_COMBINATIONS) == 3
    assert len(UNIMODAL_COMBINATIONS) == 3


def test_combination_masks():
    assert COMBINATION_MASKS[COMBINATION_RFC] == (True, True, True)
    assert COMBINATION_MASKS[COMBINATION_RF] == (True, True, False)
    assert COMBINATION_MASKS[COMBINATION_RC] == (True, False, True)
    assert COMBINATION_MASKS[COMBINATION_FC] == (False, True, True)
    assert COMBINATION_MASKS[COMBINATION_R] == (True, False, False)
    assert COMBINATION_MASKS[COMBINATION_F] == (False, True, False)
    assert COMBINATION_MASKS[COMBINATION_C] == (False, False, True)
    assert COMBINATION_MASKS[COMBINATION_EMPTY] == (False, False, False)


def test_spec_factory():
    spec_rfc = get_combination_spec(COMBINATION_RFC)
    assert spec_rfc.cardinality == 3
    assert not spec_rfc.is_empty
    assert spec_rfc.active_modalities == ("retina", "foot", "clinical")

    spec_empty = get_combination_spec(COMBINATION_EMPTY)
    assert spec_empty.cardinality == 0
    assert spec_empty.is_empty
    assert spec_empty.active_modalities == ()


def test_apply_combination_mask():
    pkt = make_dummy_packet()
    masked_rf = apply_combination_to_packet(pkt, COMBINATION_RF)

    assert masked_rf.retina.availability is True
    assert masked_rf.foot.availability is True
    assert masked_rf.clinical.availability is False
    assert masked_rf.clinical.quality == 0.0
    assert masked_rf.num_available == 2


def test_invalid_combination_id():
    with pytest.raises(ValueError):
        get_combination_spec("INVALID_COMBINATION")

    pkt = make_dummy_packet()
    with pytest.raises(ValueError):
        apply_combination_to_packet(pkt, "INVALID_COMBINATION")
