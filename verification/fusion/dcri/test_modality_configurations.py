"""
Unit tests for all seven modality availability configurations and zero-modality safe rejection (Phase C11.6).
"""

import pytest
from src.fusion.baselines.decision_packet import ModalityRecord, ControlledDecisionPacket
from src.fusion.reliability.global_reliability import (
    FROZEN_RETINA_RELIABILITY,
    FROZEN_FOOT_RELIABILITY,
    FROZEN_CLINICAL_RELIABILITY,
)
from src.fusion.dcri.dcri_engine import DCRIEngine


@pytest.fixture
def base_records():
    r = ModalityRecord("r1", "retina", 0.60, (0.5, 0.5), 0.85, 0.10, 0.90, True, FROZEN_RETINA_RELIABILITY, "v1")
    f = ModalityRecord("f1", "foot", 0.40, (0.5, 0.5), 0.75, 0.20, 0.85, True, FROZEN_FOOT_RELIABILITY, "v1")
    c = ModalityRecord("c1", "clinical", 0.25, (0.5, 0.5), 0.70, 0.15, 0.80, True, FROZEN_CLINICAL_RELIABILITY, "v1")
    return r, f, c


def _mask_packet(base_records, active_mods):
    r, f, c = base_records
    r_masked = ModalityRecord(r.sample_id, "retina", r.risk, r.calibrated_probability, r.confidence, r.uncertainty, r.quality if "retina" in active_mods else 0.0, "retina" in active_mods, r.reliability, r.model_version)
    f_masked = ModalityRecord(f.sample_id, "foot", f.risk, f.calibrated_probability, f.confidence, f.uncertainty, f.quality if "foot" in active_mods else 0.0, "foot" in active_mods, f.reliability, f.model_version)
    c_masked = ModalityRecord(c.sample_id, "clinical", c.risk, c.calibrated_probability, c.confidence, c.uncertainty, c.quality if "clinical" in active_mods else 0.0, "clinical" in active_mods, c.reliability, c.model_version)
    return ControlledDecisionPacket("PACKET_MASKED", r_masked, f_masked, c_masked, seed=115)


def test_unimodal_retina(base_records):
    engine = DCRIEngine()
    p = _mask_packet(base_records, ["retina"])
    res = engine.evaluate_packet(p, delta=0.20)
    assert res.status == "SUCCESS"
    assert res.num_active == 1
    assert res.modality_weights["retina"] == 1.0
    assert abs(res.r_fusion - 0.60) < 1e-6
    assert abs(res.dcri - (0.60 - 0.20 * 0.10)) < 1e-6


def test_unimodal_foot(base_records):
    engine = DCRIEngine()
    p = _mask_packet(base_records, ["foot"])
    res = engine.evaluate_packet(p, delta=0.20)
    assert res.status == "SUCCESS"
    assert res.num_active == 1
    assert res.modality_weights["foot"] == 1.0
    assert abs(res.r_fusion - 0.40) < 1e-6
    assert abs(res.dcri - (0.40 - 0.20 * 0.20)) < 1e-6


def test_unimodal_clinical(base_records):
    engine = DCRIEngine()
    p = _mask_packet(base_records, ["clinical"])
    res = engine.evaluate_packet(p, delta=0.20)
    assert res.status == "SUCCESS"
    assert res.num_active == 1
    assert res.modality_weights["clinical"] == 1.0
    assert abs(res.r_fusion - 0.25) < 1e-6
    assert abs(res.dcri - (0.25 - 0.20 * 0.15)) < 1e-6


def test_bimodal_configurations(base_records):
    engine = DCRIEngine()
    for active in [["retina", "foot"], ["retina", "clinical"], ["foot", "clinical"]]:
        p = _mask_packet(base_records, active)
        res = engine.evaluate_packet(p, delta=0.20)
        assert res.status == "SUCCESS"
        assert res.num_active == 2
        assert abs(sum(res.modality_weights.values()) - 1.0) < 1e-6
        for m in ["retina", "foot", "clinical"]:
            if m not in active:
                assert res.modality_weights[m] == 0.0


def test_trimodal_configuration(base_records):
    engine = DCRIEngine()
    p = _mask_packet(base_records, ["retina", "foot", "clinical"])
    res = engine.evaluate_packet(p, delta=0.20)
    assert res.status == "SUCCESS"
    assert res.num_active == 3
    assert abs(sum(res.modality_weights.values()) - 1.0) < 1e-6


def test_zero_modality_safe_rejection(base_records):
    engine = DCRIEngine()
    p = _mask_packet(base_records, [])
    res = engine.evaluate_packet(p, delta=0.20)
    assert res.status == "NO_MODALITY_AVAILABLE"
    assert res.num_active == 0
    assert res.r_fusion == 0.0
    assert res.dcri == 0.0
