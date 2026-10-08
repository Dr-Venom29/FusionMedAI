"""
src/fusion/missingness/availability_mask.py
Phase C11.8: Modality Availability Mask & Invariant Enforcement

Provides canonical availability regime definitions, packet masking transforms,
and mathematical invariant verification for missing modality robustness analysis.
"""

from typing import Dict, Tuple, Set, Optional, Sequence
import math

from src.fusion.baselines.decision_packet import ControlledDecisionPacket, ModalityRecord

# The 8 exhaustive availability regimes
ALL_REGIMES: Dict[str, Tuple[str, ...]] = {
    "tri_modal": ("retina", "foot", "clinical"),
    "retina_foot": ("retina", "foot"),
    "retina_clinical": ("retina", "clinical"),
    "foot_clinical": ("foot", "clinical"),
    "retina_only": ("retina",),
    "foot_only": ("foot",),
    "clinical_only": ("clinical",),
    "zero_modality": (),
}

# The 7 non-empty availability regimes
NON_EMPTY_REGIMES: Dict[str, Tuple[str, ...]] = {
    k: v for k, v in ALL_REGIMES.items() if k != "zero_modality"
}

# The 3 bimodal subsets (each representing removal of a single modality)
BIMODAL_REGIMES: Dict[str, Tuple[str, ...]] = {
    "retina_foot": ("retina", "foot"),        # Missing Clinical (-C)
    "retina_clinical": ("retina", "clinical"),# Missing Foot (-F)
    "foot_clinical": ("foot", "clinical"),    # Missing Retina (-R)
}

# The 3 unimodal subsets
UNIMODAL_REGIMES: Dict[str, Tuple[str, ...]] = {
    "retina_only": ("retina",),
    "foot_only": ("foot",),
    "clinical_only": ("clinical",),
}

MODALITY_NAMES: Tuple[str, ...] = ("retina", "foot", "clinical")


def apply_availability_mask(
    packet: ControlledDecisionPacket,
    active_modalities: Sequence[str],
    suffix: Optional[str] = None,
) -> ControlledDecisionPacket:
    """
    Constructs a masked copy of a ControlledDecisionPacket where only active_modalities
    have availability=True. Inactive modalities have availability=False, confidence=0.0,
    uncertainty=0.0, and quality=0.0, preserving underlying risk, calibrated probability,
    and reliability for invariant testing.
    
    Args:
        packet: Original immutable ControlledDecisionPacket.
        active_modalities: Collection of modality names that remain active.
        suffix: Optional packet_id suffix (e.g. regime name).
        
    Returns:
        New masked ControlledDecisionPacket.
    """
    active_set: Set[str] = set(active_modalities)
    records: Dict[str, ModalityRecord] = {}

    for m in MODALITY_NAMES:
        old_rec = packet.records[m]
        is_active = (m in active_set)
        records[m] = ModalityRecord(
            sample_id=old_rec.sample_id,
            modality=old_rec.modality,
            risk=old_rec.risk,
            calibrated_probability=old_rec.calibrated_probability,
            confidence=old_rec.confidence if is_active else 0.0,
            uncertainty=old_rec.uncertainty if is_active else 0.0,
            quality=old_rec.quality if is_active else 0.0,
            availability=bool(is_active),
            reliability=old_rec.reliability,
            model_version=old_rec.model_version,
        )

    pkt_id = f"{packet.packet_id}_{suffix}" if suffix else packet.packet_id
    return ControlledDecisionPacket(
        packet_id=pkt_id,
        retina=records["retina"],
        foot=records["foot"],
        clinical=records["clinical"],
        seed=packet.seed,
    )


def verify_availability_invariants(
    packet: ControlledDecisionPacket,
    weights: Dict[str, float],
    tol: float = 1e-6,
) -> Tuple[bool, Optional[str]]:
    """
    Verifies that router authority weights strictly conform to availability invariants:
    1. Unavailable modality must receive exactly zero weight: A_i = 0 => w_i = 0.0.
    2. Available modality weights must sum to 1.0 (for non-empty regimes): sum_{i in A} w_i = 1.0.
    3. Every weight must be bounded in [0.0, 1.0].
    
    Args:
        packet: Evaluated ControlledDecisionPacket.
        weights: Dictionary mapping modality names to router authority weights.
        tol: Numerical tolerance for floating-point sum verification.
        
    Returns:
        (is_valid, failure_reason)
    """
    active_mods = packet.available_modalities
    num_active = len(active_mods)

    # 1. Check all weights bounded in [0, 1]
    for m in MODALITY_NAMES:
        w = weights.get(m, 0.0)
        if math.isnan(w) or math.isinf(w):
            return False, f"Modality '{m}' weight is NaN or Inf: {w}"
        if w < -tol or w > 1.0 + tol:
            return False, f"Modality '{m}' weight {w} out of bounds [0, 1]"

    # 2. Check unavailable weight invariant: A_i = 0 => w_i = 0.0
    for m in MODALITY_NAMES:
        if m not in active_mods:
            w = weights.get(m, 0.0)
            if abs(w) > 1e-9:
                return False, f"Unavailable modality '{m}' received non-zero weight {w}"

    # 3. Check simplex sum for active modalities
    if num_active > 0:
        active_sum = sum(weights.get(m, 0.0) for m in active_mods)
        if abs(active_sum - 1.0) > tol:
            return False, f"Active modality weights sum to {active_sum:.8f}, expected 1.0"
    else:
        total_sum = sum(weights.get(m, 0.0) for m in MODALITY_NAMES)
        if abs(total_sum) > tol:
            return False, f"Zero-modality regime received non-zero total weight {total_sum:.8f}"

    return True, None
