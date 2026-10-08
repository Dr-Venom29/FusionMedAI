"""
src/fusion/missingness/dropout_scenarios.py
Phase C11.8: Deterministic & Stress Dropout Scenario Generators

Provides scenario builders for exhaustive availability regimes, sequential information loss ladders,
and targeted stress-dropout perturbations.
"""

from typing import Dict, Any, List, Tuple, Sequence, Optional
import numpy as np

from src.fusion.baselines.decision_packet import ControlledDecisionPacket
from src.fusion.missingness.availability_mask import (
    ALL_REGIMES,
    NON_EMPTY_REGIMES,
    apply_availability_mask,
    MODALITY_NAMES,
)


def get_regime_packet(
    packet: ControlledDecisionPacket,
    regime_name: str,
) -> ControlledDecisionPacket:
    """
    Constructs a masked packet for a named availability regime.
    """
    if regime_name not in ALL_REGIMES:
        raise ValueError(f"Unknown regime '{regime_name}'. Available: {list(ALL_REGIMES.keys())}")
    active_mods = ALL_REGIMES[regime_name]
    return apply_availability_mask(packet, active_mods, suffix=regime_name)


def generate_sequential_ladders(
    packet: ControlledDecisionPacket,
) -> Dict[str, List[Tuple[str, ControlledDecisionPacket]]]:
    """
    Generates sequential dropout ladders demonstrating progressive information loss.
    
    Ladders:
    1. Ladder_RetinaPrimary: RFC -> RF -> R
    2. Ladder_FootPrimary:   RFC -> FC -> F
    3. Ladder_ClinicalPrimary: RFC -> RC -> C
    """
    ladders = {
        "Ladder_RetinaPrimary": [
            ("tri_modal", get_regime_packet(packet, "tri_modal")),
            ("retina_foot", get_regime_packet(packet, "retina_foot")),
            ("retina_only", get_regime_packet(packet, "retina_only")),
        ],
        "Ladder_FootPrimary": [
            ("tri_modal", get_regime_packet(packet, "tri_modal")),
            ("foot_clinical", get_regime_packet(packet, "foot_clinical")),
            ("foot_only", get_regime_packet(packet, "foot_only")),
        ],
        "Ladder_ClinicalPrimary": [
            ("tri_modal", get_regime_packet(packet, "tri_modal")),
            ("retina_clinical", get_regime_packet(packet, "retina_clinical")),
            ("clinical_only", get_regime_packet(packet, "clinical_only")),
        ],
    }
    return ladders


def get_stress_dropped_modality(
    packet: ControlledDecisionPacket,
    stress_type: str,
) -> Tuple[str, str]:
    """
    Identifies which modality to drop based on targeted stress criteria:
    - missing_highest_confidence: argmax_{i} C_i
    - missing_lowest_confidence:  argmin_{i} C_i
    - missing_highest_reliability: argmax_{i} R_i (Retina)
    - missing_lowest_uncertainty: argmin_{i} U_i (most certain channel)
    - missing_highest_uncertainty: argmax_{i} U_i (most uncertain channel)
    - missing_lowest_quality:     argmin_{i} Q_i (poorest quality)
    - missing_highest_quality:    argmax_{i} Q_i (highest quality)
    
    Returns:
        (dropped_modality, explanation_criterion)
    """
    recs = packet.records

    if stress_type == "missing_highest_confidence":
        dropped = max(MODALITY_NAMES, key=lambda m: recs[m].confidence)
        crit = f"Highest Confidence channel (C_{dropped}={recs[dropped].confidence:.4f})"
    elif stress_type == "missing_lowest_confidence":
        dropped = min(MODALITY_NAMES, key=lambda m: recs[m].confidence)
        crit = f"Lowest Confidence channel (C_{dropped}={recs[dropped].confidence:.4f})"
    elif stress_type == "missing_highest_reliability":
        dropped = max(MODALITY_NAMES, key=lambda m: recs[m].reliability)
        crit = f"Highest Reliability prior channel (R_{dropped}={recs[dropped].reliability:.4f})"
    elif stress_type == "missing_lowest_uncertainty":
        dropped = min(MODALITY_NAMES, key=lambda m: recs[m].uncertainty)
        crit = f"Lowest Uncertainty channel (U_{dropped}={recs[dropped].uncertainty:.4f})"
    elif stress_type == "missing_highest_uncertainty":
        dropped = max(MODALITY_NAMES, key=lambda m: recs[m].uncertainty)
        crit = f"Highest Uncertainty channel (U_{dropped}={recs[dropped].uncertainty:.4f})"
    elif stress_type == "missing_lowest_quality":
        dropped = min(MODALITY_NAMES, key=lambda m: recs[m].quality)
        crit = f"Lowest Quality channel (Q_{dropped}={recs[dropped].quality:.4f})"
    elif stress_type == "missing_highest_quality":
        dropped = max(MODALITY_NAMES, key=lambda m: recs[m].quality)
        crit = f"Highest Quality channel (Q_{dropped}={recs[dropped].quality:.4f})"
    else:
        raise ValueError(f"Unknown stress_type '{stress_type}'")

    return dropped, crit


def get_stress_packet(
    packet: ControlledDecisionPacket,
    stress_type: str,
) -> Tuple[ControlledDecisionPacket, str, str]:
    """
    Constructs a stress-dropped ControlledDecisionPacket.
    
    Returns:
        (stress_packet, dropped_modality, criterion_description)
    """
    dropped, crit = get_stress_dropped_modality(packet, stress_type)
    active = tuple(m for m in MODALITY_NAMES if m != dropped)
    stress_pkt = apply_availability_mask(packet, active, suffix=stress_type)
    return stress_pkt, dropped, crit


def get_random_dropout_packet(
    packet: ControlledDecisionPacket,
    rng: np.random.Generator,
) -> Tuple[ControlledDecisionPacket, str]:
    """
    Samples a random non-empty availability regime reproducibly.
    """
    regime_names = list(NON_EMPTY_REGIMES.keys())
    chosen_regime = str(rng.choice(regime_names))
    active = NON_EMPTY_REGIMES[chosen_regime]
    return apply_availability_mask(packet, active, suffix=f"rnd_{chosen_regime}"), chosen_regime
