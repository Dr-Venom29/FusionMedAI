"""
src/fusion/combination_analysis/combination_definition.py
Phase C11.9: Modality Combination Definitions & Immutables

Defines the 8 canonical modality combinations across the 3 diagnostic modalities:
Retina (R), Foot (F), Clinical (C).
Strictly fail-closed: EMPTY represents no modality availability and cannot produce a fused risk.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Set
import numpy as np

from src.fusion.baselines.decision_packet import ControlledDecisionPacket, ModalityRecord


# Canonical combination identifiers
COMBINATION_RFC = "RFC"      # Tri-modal (1, 1, 1)
COMBINATION_RF = "RF"        # Bimodal Retina + Foot (1, 1, 0)
COMBINATION_RC = "RC"        # Bimodal Retina + Clinical (1, 0, 1)
COMBINATION_FC = "FC"        # Bimodal Foot + Clinical (0, 1, 1)
COMBINATION_R = "R"          # Unimodal Retina (1, 0, 0)
COMBINATION_F = "F"          # Unimodal Foot (0, 1, 0)
COMBINATION_C = "C"          # Unimodal Clinical (0, 0, 1)
COMBINATION_EMPTY = "EMPTY"  # Zero modality (0, 0, 0)

ALL_COMBINATIONS: Tuple[str, ...] = (
    COMBINATION_RFC,
    COMBINATION_RF,
    COMBINATION_RC,
    COMBINATION_FC,
    COMBINATION_R,
    COMBINATION_F,
    COMBINATION_C,
    COMBINATION_EMPTY,
)

NON_EMPTY_COMBINATIONS: Tuple[str, ...] = (
    COMBINATION_RFC,
    COMBINATION_RF,
    COMBINATION_RC,
    COMBINATION_FC,
    COMBINATION_R,
    COMBINATION_F,
    COMBINATION_C,
)

BIMODAL_COMBINATIONS: Tuple[str, ...] = (
    COMBINATION_RF,
    COMBINATION_RC,
    COMBINATION_FC,
)

UNIMODAL_COMBINATIONS: Tuple[str, ...] = (
    COMBINATION_R,
    COMBINATION_F,
    COMBINATION_C,
)

# Availability mask lookup: (A_R, A_F, A_C)
COMBINATION_MASKS: Dict[str, Tuple[bool, bool, bool]] = {
    COMBINATION_RFC: (True, True, True),
    COMBINATION_RF: (True, True, False),
    COMBINATION_RC: (True, False, True),
    COMBINATION_FC: (False, True, True),
    COMBINATION_R: (True, False, False),
    COMBINATION_F: (False, True, False),
    COMBINATION_C: (False, False, True),
    COMBINATION_EMPTY: (False, False, False),
}

# Modality active channel mapping
COMBINATION_ACTIVE_MODALITIES: Dict[str, Tuple[str, ...]] = {
    COMBINATION_RFC: ("retina", "foot", "clinical"),
    COMBINATION_RF: ("retina", "foot"),
    COMBINATION_RC: ("retina", "clinical"),
    COMBINATION_FC: ("foot", "clinical"),
    COMBINATION_R: ("retina",),
    COMBINATION_F: ("foot",),
    COMBINATION_C: ("clinical",),
    COMBINATION_EMPTY: (),
}

COMBINATION_CARDINALITY: Dict[str, int] = {
    COMBINATION_RFC: 3,
    COMBINATION_RF: 2,
    COMBINATION_RC: 2,
    COMBINATION_FC: 2,
    COMBINATION_R: 1,
    COMBINATION_F: 1,
    COMBINATION_C: 1,
    COMBINATION_EMPTY: 0,
}


@dataclass(frozen=True)
class ModalityCombinationSpec:
    """Immutable specification of a single modality combination."""
    combination_id: str
    availability_retina: bool
    availability_foot: bool
    availability_clinical: bool
    active_modalities: Tuple[str, ...]
    cardinality: int
    is_empty: bool

    @classmethod
    def from_id(cls, comb_id: str) -> "ModalityCombinationSpec":
        if comb_id not in COMBINATION_MASKS:
            raise ValueError(f"Unknown combination ID '{comb_id}'. Allowed: {ALL_COMBINATIONS}")
        mask = COMBINATION_MASKS[comb_id]
        active = COMBINATION_ACTIVE_MODALITIES[comb_id]
        card = COMBINATION_CARDINALITY[comb_id]
        return cls(
            combination_id=comb_id,
            availability_retina=mask[0],
            availability_foot=mask[1],
            availability_clinical=mask[2],
            active_modalities=active,
            cardinality=card,
            is_empty=(card == 0),
        )


def get_combination_spec(comb_id: str) -> ModalityCombinationSpec:
    """Returns immutable spec for a combination identifier."""
    return ModalityCombinationSpec.from_id(comb_id)


def apply_combination_to_packet(
    packet: ControlledDecisionPacket,
    comb_id: str,
) -> ControlledDecisionPacket:
    """
    Applies the availability mask of comb_id to an existing ControlledDecisionPacket.
    Unavailable channels have availability=False, quality=0.0.
    """
    if comb_id not in COMBINATION_MASKS:
        raise ValueError(f"Invalid combination_id: '{comb_id}'.")

    a_r, a_f, a_c = COMBINATION_MASKS[comb_id]

    r_mod = ModalityRecord(
        sample_id=packet.retina.sample_id,
        modality="retina",
        risk=packet.retina.risk,
        calibrated_probability=packet.retina.calibrated_probability,
        confidence=packet.retina.confidence,
        uncertainty=packet.retina.uncertainty,
        quality=packet.retina.quality if a_r else 0.0,
        availability=a_r,
        reliability=packet.retina.reliability,
        model_version=packet.retina.model_version,
    )
    f_mod = ModalityRecord(
        sample_id=packet.foot.sample_id,
        modality="foot",
        risk=packet.foot.risk,
        calibrated_probability=packet.foot.calibrated_probability,
        confidence=packet.foot.confidence,
        uncertainty=packet.foot.uncertainty,
        quality=packet.foot.quality if a_f else 0.0,
        availability=a_f,
        reliability=packet.foot.reliability,
        model_version=packet.foot.model_version,
    )
    c_mod = ModalityRecord(
        sample_id=packet.clinical.sample_id,
        modality="clinical",
        risk=packet.clinical.risk,
        calibrated_probability=packet.clinical.calibrated_probability,
        confidence=packet.clinical.confidence,
        uncertainty=packet.clinical.uncertainty,
        quality=packet.clinical.quality if a_c else 0.0,
        availability=a_c,
        reliability=packet.clinical.reliability,
        model_version=packet.clinical.model_version,
    )

    return ControlledDecisionPacket(
        packet_id=f"{packet.packet_id}_{comb_id}",
        retina=r_mod,
        foot=f_mod,
        clinical=c_mod,
        seed=packet.seed,
    )
