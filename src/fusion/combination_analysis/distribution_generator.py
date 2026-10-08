"""
src/fusion/combination_analysis/distribution_generator.py
Phase C11.9: Controlled Modality-Combination Frequency Distributions

Defines the pre-registered combination distributions:
1. D1_BALANCED: Equal distribution across 7 non-empty combinations (~14.286% each).
2. D2_MODERATE_HEAD_TAIL: Moderate power-law dropoff (RFC: 35%, RF: 25%, RC: 15%, FC: 10%, R: 6%, F: 5%, C: 4%).
3. D3_STRONG_LONG_TAIL: Steep long-tail distribution (RFC: 50%, RF: 25%, RC: 10%, FC: 8%, R: 4%, F: 2%, C: 1%).

Provides deterministic, reproducible cohort assignment routines.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Sequence, Optional
import numpy as np

from src.fusion.baselines.decision_packet import ControlledDecisionPacket
from src.fusion.combination_analysis.combination_definition import (
    NON_EMPTY_COMBINATIONS,
    COMBINATION_RFC,
    COMBINATION_RF,
    COMBINATION_RC,
    COMBINATION_FC,
    COMBINATION_R,
    COMBINATION_F,
    COMBINATION_C,
    apply_combination_to_packet,
)


DISTRIBUTION_D1_BALANCED = "D1_BALANCED"
DISTRIBUTION_D2_MODERATE_HEAD_TAIL = "D2_MODERATE_HEAD_TAIL"
DISTRIBUTION_D3_STRONG_LONG_TAIL = "D3_STRONG_LONG_TAIL"

ALL_DISTRIBUTIONS = (
    DISTRIBUTION_D1_BALANCED,
    DISTRIBUTION_D2_MODERATE_HEAD_TAIL,
    DISTRIBUTION_D3_STRONG_LONG_TAIL,
)

# Canonical pre-registered probability vectors over NON_EMPTY_COMBINATIONS
PROBABILITY_VECTORS: Dict[str, Dict[str, float]] = {
    DISTRIBUTION_D1_BALANCED: {
        COMBINATION_RFC: 1.0 / 7.0,
        COMBINATION_RF:  1.0 / 7.0,
        COMBINATION_RC:  1.0 / 7.0,
        COMBINATION_FC:  1.0 / 7.0,
        COMBINATION_R:   1.0 / 7.0,
        COMBINATION_F:   1.0 / 7.0,
        COMBINATION_C:   1.0 / 7.0,
    },
    DISTRIBUTION_D2_MODERATE_HEAD_TAIL: {
        COMBINATION_RFC: 0.35,
        COMBINATION_RF:  0.25,
        COMBINATION_RC:  0.15,
        COMBINATION_FC:  0.10,
        COMBINATION_R:   0.06,
        COMBINATION_F:   0.05,
        COMBINATION_C:   0.04,
    },
    DISTRIBUTION_D3_STRONG_LONG_TAIL: {
        COMBINATION_RFC: 0.50,
        COMBINATION_RF:  0.25,
        COMBINATION_RC:  0.10,
        COMBINATION_FC:  0.08,
        COMBINATION_R:   0.04,
        COMBINATION_F:   0.02,
        COMBINATION_C:   0.01,
    },
}


@dataclass(frozen=True)
class DistributionConfig:
    """Immutable specification of a controlled frequency distribution."""
    distribution_id: str
    name: str
    description: str
    probabilities: Dict[str, float]
    expected_counts_n500: Dict[str, int]

    def __post_init__(self) -> None:
        total_p = sum(self.probabilities.values())
        if not np.isclose(total_p, 1.0, atol=1e-6):
            raise ValueError(f"Probabilities must sum to 1.0, got {total_p:.6f}")
        for c, p in self.probabilities.items():
            if p <= 0.0 or p > 1.0:
                raise ValueError(f"Invalid probability for {c}: {p}")


def get_distribution_config(dist_id: str) -> DistributionConfig:
    """Returns the immutable distribution config for a registered distribution ID."""
    if dist_id not in PROBABILITY_VECTORS:
        raise ValueError(f"Unknown distribution ID '{dist_id}'. Allowed: {ALL_DISTRIBUTIONS}")

    p_dict = PROBABILITY_VECTORS[dist_id]

    if dist_id == DISTRIBUTION_D1_BALANCED:
        name = "D1: Balanced Distribution"
        desc = "Uniform probability distribution across all 7 non-empty combinations (~14.286% each)."
        # 500 = 71 * 5 + 72 * 2 = 355 + 144 = 499 + 1 => 71 * 4 + 72 * 3 = 284 + 216 = 500
        # Deterministic count partition
        counts = {
            COMBINATION_RFC: 72,
            COMBINATION_RF:  72,
            COMBINATION_RC:  72,
            COMBINATION_FC:  71,
            COMBINATION_R:   71,
            COMBINATION_F:   71,
            COMBINATION_C:   71,
        }
    elif dist_id == DISTRIBUTION_D2_MODERATE_HEAD_TAIL:
        name = "D2: Moderate Head-Tail Distribution"
        desc = "Moderate power-law dropoff favoring multi-modal combinations (Head: 60%, Middle: 25%, Tail: 15%)."
        counts = {
            COMBINATION_RFC: 175,  # 35%
            COMBINATION_RF:  125,  # 25%
            COMBINATION_RC:  75,   # 15%
            COMBINATION_FC:  50,   # 10%
            COMBINATION_R:   30,   # 6%
            COMBINATION_F:   25,   # 5%
            COMBINATION_C:   20,   # 4%
        }
    elif dist_id == DISTRIBUTION_D3_STRONG_LONG_TAIL:
        name = "D3: Strong Long-Tail Distribution"
        desc = "Steep long-tail distribution with dominant tri-modal head and rare unimodal tail (Head: 75%, Middle: 18%, Tail: 7%)."
        counts = {
            COMBINATION_RFC: 250,  # 50%
            COMBINATION_RF:  125,  # 25%
            COMBINATION_RC:  50,   # 10%
            COMBINATION_FC:  40,   # 8%
            COMBINATION_R:   20,   # 4%
            COMBINATION_F:   10,   # 2%
            COMBINATION_C:   5,    # 1%
        }
    else:
        raise ValueError(f"Unhandled distribution_id {dist_id}")

    return DistributionConfig(
        distribution_id=dist_id,
        name=name,
        description=desc,
        probabilities=p_dict,
        expected_counts_n500=counts,
    )


@dataclass(frozen=True)
class AssignedPacket:
    """An individual packet assigned to a specific combination within a distribution."""
    original_packet_id: str
    assigned_combination: str
    packet: ControlledDecisionPacket
    distribution_id: str
    sample_index: int


def generate_stratified_distribution_cohort(
    cohort: Sequence[ControlledDecisionPacket],
    dist_id: str,
    seed: int = 115,
) -> List[AssignedPacket]:
    """
    Deterministically assigns packets from the frozen cohort to combinations according
    to the target distribution's exact target counts.
    
    Args:
        cohort: Sequence of N ControlledDecisionPackets (e.g., N=500).
        dist_id: Distribution identifier (D1, D2, or D3).
        seed: Random seed for deterministic assignment permutation.
        
    Returns:
        List of AssignedPacket objects containing the masked packets.
    """
    config = get_distribution_config(dist_id)
    n_total = len(cohort)

    if n_total == 500:
        counts = config.expected_counts_n500
    else:
        # Largest Remainder (Hare-Niemeyer) method for exact integer allocation summing to n_total
        raw_shares = {c: config.probabilities[c] * n_total for c in NON_EMPTY_COMBINATIONS}
        base_counts = {c: int(raw_shares[c]) for c in NON_EMPTY_COMBINATIONS}
        remainders = {c: raw_shares[c] - base_counts[c] for c in NON_EMPTY_COMBINATIONS}
        leftover = n_total - sum(base_counts.values())
        canonical_order = {c: i for i, c in enumerate(NON_EMPTY_COMBINATIONS)}
        sorted_by_rem = sorted(NON_EMPTY_COMBINATIONS, key=lambda c: (-remainders[c], canonical_order[c]))
        counts = dict(base_counts)
        for i in range(leftover):
            counts[sorted_by_rem[i]] += 1

    # Deterministic permutation of packet indices
    rng = np.random.RandomState(seed)
    indices = np.arange(n_total)
    permuted_indices = rng.permutation(indices)

    assigned: List[AssignedPacket] = []
    curr_offset = 0

    for comb_id in NON_EMPTY_COMBINATIONS:
        c_count = counts[comb_id]
        comb_indices = permuted_indices[curr_offset : curr_offset + c_count]
        curr_offset += c_count

        for idx in comb_indices:
            orig_pkt = cohort[idx]
            masked_pkt = apply_combination_to_packet(orig_pkt, comb_id)
            assigned.append(
                AssignedPacket(
                    original_packet_id=orig_pkt.packet_id,
                    assigned_combination=comb_id,
                    packet=masked_pkt,
                    distribution_id=dist_id,
                    sample_index=int(idx),
                )
            )

    return assigned
