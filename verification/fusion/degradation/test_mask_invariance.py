"""
verification/fusion/degradation/test_mask_invariance.py
Phase C11.10: Verification of Hard-Mask Invariance Under Unavailable Modality Degradation
"""

import pytest
from pathlib import Path
from src.fusion.dcri.cohort_loader import load_frozen_cohort
from src.fusion.degradation.degradation_engine import DegradationEngine, apply_degradation_to_packet
from src.fusion.degradation.degradation_spec import (
    SEVERITY_D3_SEVERE,
    MODALITY_RETINA,
    RETINA_OPERATORS,
)
from src.fusion.baselines.decision_packet import ControlledDecisionPacket, ModalityRecord
from src.fusion.degradation.response_metrics import check_hard_mask_invariance

repo_root = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def engine():
    cohort = load_frozen_cohort(repo_root=repo_root, n_packets=500, seed=115)
    return DegradationEngine(cohort=cohort, n_bootstrap=100, seed=115)


def test_unavailable_modality_corruption_invariance(engine):
    """
    If Retina is unavailable (A_R = False), degrading or corrupting its fields
    must have ZERO effect on active routing weights or fused risk.
    """
    pkt = engine.cohort[0]
    
    # Create packet where retina is unavailable
    unavail_retina = ModalityRecord(
        sample_id=pkt.retina.sample_id,
        modality="retina",
        risk=pkt.retina.risk,
        calibrated_probability=pkt.retina.calibrated_probability,
        confidence=pkt.retina.confidence,
        uncertainty=pkt.retina.uncertainty,
        quality=0.0,
        availability=False,
        reliability=pkt.retina.reliability,
        model_version=pkt.retina.model_version,
    )
    clean_unavail_pkt = ControlledDecisionPacket(
        packet_id=pkt.packet_id,
        retina=unavail_retina,
        foot=pkt.foot,
        clinical=pkt.clinical,
        seed=pkt.seed,
    )

    # Degrade retina on this packet
    deg_unavail_pkt = apply_degradation_to_packet(
        clean_unavail_pkt, MODALITY_RETINA, RETINA_OPERATORS[0], SEVERITY_D3_SEVERE
    )

    # Evaluate both
    res_clean = engine.dcri_engine.evaluate_packet(clean_unavail_pkt, delta=0.20)
    res_deg = engine.dcri_engine.evaluate_packet(deg_unavail_pkt, delta=0.20)

    # Invariance check
    is_invariant = check_hard_mask_invariance(
        clean_unavail_pkt,
        deg_unavail_pkt,
        res_clean.modality_weights,
        res_deg.modality_weights,
        res_clean.r_fusion,
        res_deg.r_fusion,
    )
    assert is_invariant is True
    assert res_clean.modality_weights["retina"] == 0.0
    assert res_deg.modality_weights["retina"] == 0.0
    assert res_clean.r_fusion == pytest.approx(res_deg.r_fusion, abs=1e-7)
