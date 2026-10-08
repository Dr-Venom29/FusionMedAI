"""
src/fusion/missingness/stress_tests.py
Phase C11.8: Behavioral Stress Tests & Masked-Value Invariance Verification

Provides tests for:
1. Masked-Value Invariance: Proves that mutating unavailable channel signals cannot alter active outputs.
2. Availability vs Low Quality: Proves that A_i=0 (hard zero weight) is distinct from A_i=1, Q_i=0.
3. Targeted Stress Dropouts: Evaluates authority and risk shifts under worst-case/best-case channel loss.
"""

from typing import Dict, Any, List, Tuple, Sequence, Optional
import math
import numpy as np

from src.fusion.baselines.decision_packet import ControlledDecisionPacket, ModalityRecord
from src.fusion.dcri.dcri_engine import DCRIEngine
from src.fusion.missingness.availability_mask import (
    apply_availability_mask,
    MODALITY_NAMES,
)
from src.fusion.missingness.dropout_scenarios import (
    get_stress_packet,
)
from src.fusion.missingness.missingness_result import (
    MaskedInvarianceRecord,
    StressScenarioRecord,
)


def evaluate_masked_value_invariance(
    packet: ControlledDecisionPacket,
    dcri_engine: DCRIEngine,
    delta: float = 0.20,
) -> List[MaskedInvarianceRecord]:
    """
    Critical Security/Integrity Invariant:
    When modality j is unavailable (A_j = 0), mutating any of its internal values
    (risk, confidence, uncertainty, quality, reliability) MUST NOT alter the active
    authority weights, fused risk R_fusion, or DCRI_delta.
    """
    records: List[MaskedInvarianceRecord] = []

    # Test each bimodal configuration where exactly 1 modality is unavailable
    bimodal_pairs = [
        ("retina", ("foot", "clinical")),
        ("foot", ("retina", "clinical")),
        ("clinical", ("retina", "foot")),
    ]

    perturbation_sets = [
        ("extreme_high_risk_max_confidence", {"risk": 1.0, "confidence": 1.0, "quality": 0.0, "uncertainty": 0.0}),
        ("extreme_zero_risk_zero_confidence", {"risk": 0.0, "confidence": 0.0, "quality": 0.0, "uncertainty": 1.0}),
        ("extreme_high_uncertainty_mid_risk", {"risk": 0.5, "confidence": 0.5, "quality": 0.0, "uncertainty": 1.0}),
        ("near_one_risk_low_uncertainty", {"risk": 0.999, "confidence": 0.999, "quality": 0.0, "uncertainty": 0.001}),
        ("near_zero_risk_high_uncertainty", {"risk": 0.001, "confidence": 0.001, "quality": 0.0, "uncertainty": 0.999}),
    ]

    for unavail_mod, active_mods in bimodal_pairs:
        # 1. Base masked packet
        base_pkt = apply_availability_mask(packet, active_mods, suffix="base_inv")
        base_res = dcri_engine.evaluate_packet(base_pkt, delta=delta)

        for p_name, p_vals in perturbation_sets:
            # 2. Construct corrupted copy where unavailable modality has perturbed values
            corrupted_records = {}
            for m in MODALITY_NAMES:
                old_m = base_pkt.records[m]
                if m == unavail_mod:
                    corrupted_records[m] = ModalityRecord(
                        sample_id=old_m.sample_id,
                        modality=old_m.modality,
                        risk=p_vals["risk"],
                        calibrated_probability=tuple([p_vals["risk"], 1.0 - p_vals["risk"]] if old_m.modality == "clinical" else [p_vals["risk"]] + [0.0]*(len(old_m.calibrated_probability)-1)),
                        confidence=p_vals["confidence"],
                        uncertainty=p_vals["uncertainty"],
                        quality=p_vals["quality"],
                        availability=False,  # Hard unavailable
                        reliability=old_m.reliability,  # Frozen prior
                        model_version=old_m.model_version,
                    )
                else:
                    corrupted_records[m] = old_m

            corrupted_pkt = ControlledDecisionPacket(
                packet_id=f"{packet.packet_id}_corrupt_{unavail_mod}_{p_name}",
                retina=corrupted_records["retina"],
                foot=corrupted_records["foot"],
                clinical=corrupted_records["clinical"],
                seed=packet.seed,
            )

            corrupt_res = dcri_engine.evaluate_packet(corrupted_pkt, delta=delta)

            # Compare active weights, r_fusion, dcri
            max_w_diff = max(
                abs(base_res.modality_weights.get(m, 0.0) - corrupt_res.modality_weights.get(m, 0.0))
                for m in active_mods
            )
            r_diff = abs(base_res.r_fusion - corrupt_res.r_fusion)
            dcri_diff = abs(base_res.dcri - corrupt_res.dcri)

            passed = (max_w_diff < 1e-9 and r_diff < 1e-9 and dcri_diff < 1e-9)

            records.append(
                MaskedInvarianceRecord(
                    packet_id=packet.packet_id,
                    unavailable_modality=unavail_mod,
                    perturbation_type=p_name,
                    active_weights_match=(max_w_diff < 1e-9),
                    r_fusion_match=(r_diff < 1e-9),
                    dcri_match=(dcri_diff < 1e-9),
                    max_weight_diff=round(float(max_w_diff), 9),
                    r_fusion_diff=round(float(r_diff), 9),
                    dcri_diff=round(float(dcri_diff), 9),
                    passed=passed,
                )
            )

    return records


def evaluate_unavailable_vs_low_quality(
    packet: ControlledDecisionPacket,
    dcri_engine: DCRIEngine,
    delta: float = 0.20,
) -> Dict[str, Any]:
    """
    Verifies the fundamental distinction between missingness and low quality:
    - Unavailable (A_i = 0): w_i = 0.0 strictly.
    - Low Quality (A_i = 1, Q_i = 0.0): w_i >= 0 (participates in router softmax, modulated by quality weight).
    """
    # 1. Full tri-modal base
    full_res = dcri_engine.evaluate_packet(packet, delta=delta)

    # 2. Unavailable Retina (A_R = 0)
    unavail_pkt = apply_availability_mask(packet, ("foot", "clinical"), suffix="unavail_R")
    unavail_res = dcri_engine.evaluate_packet(unavail_pkt, delta=delta)

    # 3. Degraded Quality Retina (A_R = 1, Q_R = 0.0, U_R = 0.99, C_R = 0.01)
    degraded_records = {
        "retina": ModalityRecord(
            sample_id=packet.retina.sample_id,
            modality="retina",
            risk=packet.retina.risk,
            calibrated_probability=packet.retina.calibrated_probability,
            confidence=0.01,
            uncertainty=0.99,
            quality=0.0,
            availability=True,  # Still technically available
            reliability=packet.retina.reliability,
            model_version=packet.retina.model_version,
        ),
        "foot": packet.foot,
        "clinical": packet.clinical,
    }
    degraded_pkt = ControlledDecisionPacket(
        packet_id=f"{packet.packet_id}_degraded_R",
        retina=degraded_records["retina"],
        foot=degraded_records["foot"],
        clinical=degraded_records["clinical"],
        seed=packet.seed,
    )
    degraded_res = dcri_engine.evaluate_packet(degraded_pkt, delta=delta)

    return {
        "unavailable_retina_weight": unavail_res.modality_weights.get("retina", 0.0),
        "degraded_retina_weight": degraded_res.modality_weights.get("retina", 0.0),
        "full_retina_weight": full_res.modality_weights.get("retina", 0.0),
        "distinction_verified": (
            unavail_res.modality_weights.get("retina", 0.0) == 0.0
            and degraded_res.modality_weights.get("retina", 0.0) > 0.0
        ),
    }


def evaluate_stress_scenarios(
    packet: ControlledDecisionPacket,
    dcri_engine: DCRIEngine,
    delta: float = 0.20,
) -> Dict[str, StressScenarioRecord]:
    """
    Evaluates all 7 targeted stress dropout scenarios on a single packet.
    """
    stress_types = [
        "missing_highest_confidence",
        "missing_lowest_confidence",
        "missing_highest_reliability",
        "missing_lowest_uncertainty",
        "missing_highest_uncertainty",
        "missing_lowest_quality",
        "missing_highest_quality",
    ]

    full_res = dcri_engine.evaluate_packet(packet, delta=delta)
    results: Dict[str, StressScenarioRecord] = {}

    for st in stress_types:
        stress_pkt, dropped_mod, crit = get_stress_packet(packet, st)
        stress_res = dcri_engine.evaluate_packet(stress_pkt, delta=delta)

        delta_r = abs(full_res.r_fusion - stress_res.r_fusion)
        delta_dcri = abs(full_res.dcri - stress_res.dcri)

        results[st] = StressScenarioRecord(
            scenario_name=st,
            criterion=crit,
            dropped_modality=dropped_mod,
            remaining_modalities=tuple(m for m in MODALITY_NAMES if m != dropped_mod),
            r_fusion_full=round(float(full_res.r_fusion), 6),
            r_fusion_stress=round(float(stress_res.r_fusion), 6),
            delta_r=round(float(delta_r), 6),
            dcri_full=round(float(full_res.dcri), 6),
            dcri_stress=round(float(stress_res.dcri), 6),
            delta_dcri=round(float(delta_dcri), 6),
            weights_full={m: round(float(full_res.modality_weights.get(m, 0.0)), 6) for m in MODALITY_NAMES},
            weights_stress={m: round(float(stress_res.modality_weights.get(m, 0.0)), 6) for m in MODALITY_NAMES},
        )

    return results
