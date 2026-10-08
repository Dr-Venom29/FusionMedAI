"""
src/fusion/combination_analysis/calibration_analysis.py
Phase C11.9: Modality Calibration Reporting & Methodological Boundary

Documents and reports the modality-level calibrated probability characteristics across
participating combinations, strictly enforcing that decision-level fused risk R_fusion
and DCRI are derived decision indices rather than clinically calibrated probabilities.
"""

from typing import Dict, Any, List, Sequence
import numpy as np

from src.fusion.combination_analysis.combination_definition import (
    NON_EMPTY_COMBINATIONS,
    COMBINATION_MASKS,
)
from src.fusion.combination_analysis.distribution_generator import AssignedPacket


def analyze_modality_calibration_availability(
    assigned_cohort: Sequence[AssignedPacket],
) -> Dict[str, Any]:
    """
    Evaluates calibrated probability properties of participating modalities
    within each modality combination.
    """
    by_combination: Dict[str, Dict[str, Any]] = {
        c: {
            "retina_calibrated_p_mean": None,
            "foot_calibrated_p_mean": None,
            "clinical_calibrated_p_mean": None,
            "n_packets": 0,
        }
        for c in NON_EMPTY_COMBINATIONS
    }

    retina_probs: Dict[str, List[float]] = {c: [] for c in NON_EMPTY_COMBINATIONS}
    foot_probs: Dict[str, List[float]] = {c: [] for c in NON_EMPTY_COMBINATIONS}
    clinical_probs: Dict[str, List[float]] = {c: [] for c in NON_EMPTY_COMBINATIONS}

    for ap in assigned_cohort:
        c = ap.assigned_combination
        pkt = ap.packet
        mask = COMBINATION_MASKS[c]

        by_combination[c]["n_packets"] += 1
        if mask[0]:  # Retina available
            # Max calibrated class probability
            retina_probs[c].append(float(np.max(pkt.retina.calibrated_probability)))
        if mask[1]:  # Foot available
            foot_probs[c].append(float(np.max(pkt.foot.calibrated_probability)))
        if mask[2]:  # Clinical available
            clinical_probs[c].append(float(np.max(pkt.clinical.calibrated_probability)))

    for c in NON_EMPTY_COMBINATIONS:
        if retina_probs[c]:
            by_combination[c]["retina_calibrated_p_mean"] = float(np.mean(retina_probs[c]))
        if foot_probs[c]:
            by_combination[c]["foot_calibrated_p_mean"] = float(np.mean(foot_probs[c]))
        if clinical_probs[c]:
            by_combination[c]["clinical_calibrated_p_mean"] = float(np.mean(clinical_probs[c]))

    return {
        "modality_calibration_by_combination": by_combination,
        "methodological_declaration": (
            "Modality-level probabilities are calibrated independently using validation-derived scaling "
            "(Temperature Scaling for Retina, Vector Scaling for Foot, Isotonic Regression for Clinical). "
            "R_fusion and DCRI are derived decision indices; no clinical ground truth is fabricated for fusion-level ECE."
        ),
    }
