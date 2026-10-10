"""
src/fusion/combination_analysis/distribution_generator.py
Phase C11.9: Controlled Modality-Combination Frequency Distributions

Defines the pre-registered combination distributions:
1. D1_BALANCED: Equal distribution across 7 non-empty combinations (~14.286% each).
2. D2_MODERATE_HEAD_TAIL: Moderate power-law dropoff (RFC: 35%, RF: 25%, RC: 15%, FC: 10%, R: 6%, F: 5%, C: 4%).
3. D3_STRONG_LONG_TAIL: Steep long-tail distribution (RFC: 50%, RF: 25%, RC: 10%, FC: 8%, R: 4%, F: 2%, C: 1%).

Methodological & Statistical Clarifications:
- Cohort Generation vs Combination Assignment: Cohort generation (generate_controlled_synthetic_cohort)
  creates fully observed tri-modal decision packets (RFC) with simulated latent risk and modality records.
  Stratified combination assignment (generate_stratified_distribution_cohort) subsequently applies
  the availability masks of D1, D2, or D3 to those packets.
- Independence Scope: The 30 seeds produce independently sampled synthetic cohorts from the specified
  generative model. They represent controlled Monte Carlo simulation cohorts, not 30 independent clinical populations.
- Tail Metric Weighting: Tail sensitivity D_tail is evaluated at both the packet level (micro-average,
  weighting each tail packet equally, N_tail=35 in D3) and combination level (macro-average, weighting
  each tail combination {R, F, C} equally).
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


def _generate_multiclass_probabilities(risk: float, num_classes: int, rng: np.random.RandomState) -> Tuple[float, ...]:
    """Generates a realistic calibrated probability distribution peaking around risk level."""
    center = float(risk) * (num_classes - 1)
    indices = np.arange(num_classes, dtype=np.float64)
    logits = -0.5 * ((indices - center) ** 2) / (0.85 ** 2)
    logits += rng.normal(0.0, 0.12, size=num_classes)
    exp_l = np.exp(logits - np.max(logits))
    probs = exp_l / np.sum(exp_l)
    rounded = [round(float(p), 6) for p in probs]
    rounded[-1] = round(1.0 - sum(rounded[:-1]), 6)
    return tuple(rounded)


def generate_controlled_synthetic_cohort(
    seed: int,
    n_packets: int = 500,
    cohort_prefix: str = "PKT_COMB",
) -> List[ControlledDecisionPacket]:
    """
    Generates an independent cohort of n_packets fully populated ControlledDecisionPackets
    with all 3 modalities active (tri-modal RFC state) ready for stratified combination masking.
    
    Generative process:
    - Samples latent risk Y* ~ Beta(2, 2) on [0.02, 0.98].
    - Generates modality risks with calibrated errors matching empirical reliability priors:
      Retina (sigma=0.075), Foot (sigma=0.095), Clinical (sigma=0.140).
    - Samples realistic calibrated posterior probabilities, confidence, uncertainty, and quality.
    - Preserves exact immutable contracts and frozen reliability constants.
    """
    from src.fusion.baselines.decision_packet import ModalityRecord
    from src.fusion.reliability.global_reliability import (
        FROZEN_RETINA_RELIABILITY,
        FROZEN_FOOT_RELIABILITY,
        FROZEN_CLINICAL_RELIABILITY,
    )

    rng = np.random.RandomState(int(seed))
    packets: List[ControlledDecisionPacket] = []

    for idx in range(n_packets):
        packet_id = f"{cohort_prefix}_{seed}_{idx:04d}"

        # Latent continuous risk
        raw_y = float(rng.beta(2.0, 2.0))
        y_star = round(float(np.clip(raw_y, 0.02, 0.98)), 6)

        # Retina Modality (5-class ordinal classification)
        r_err = float(rng.normal(0.0, 0.075))
        r_risk = round(float(np.clip(y_star + r_err, 0.0, 1.0)), 6)
        r_probs = _generate_multiclass_probabilities(r_risk, num_classes=5, rng=rng)
        r_conf = round(float(np.max(r_probs)), 6)
        r_unc = round(float(np.clip(abs(r_err) * 1.5 + rng.uniform(0.01, 0.08), 0.001, 0.45)), 6)
        r_qual = round(float(rng.uniform(0.85, 0.99)), 4)

        retina_rec = ModalityRecord(
            sample_id=f"SAMP_{packet_id}_RET",
            modality="retina",
            risk=r_risk,
            calibrated_probability=r_probs,
            confidence=r_conf,
            uncertainty=r_unc,
            quality=r_qual,
            availability=True,
            reliability=FROZEN_RETINA_RELIABILITY,
            model_version="retina_efficientnet_b3_v1.0",
        )

        # Foot Modality (4-class classification)
        f_err = float(rng.normal(0.0, 0.095))
        f_risk = round(float(np.clip(y_star + f_err, 0.0, 1.0)), 6)
        f_probs = _generate_multiclass_probabilities(f_risk, num_classes=4, rng=rng)
        f_conf = round(float(np.max(f_probs)), 6)
        f_unc = round(float(np.clip(abs(f_err) * 1.5 + rng.uniform(0.01, 0.10), 0.001, 0.55)), 6)
        f_qual = round(float(rng.uniform(0.80, 0.98)), 4)

        foot_rec = ModalityRecord(
            sample_id=f"SAMP_{packet_id}_FOO",
            modality="foot",
            risk=f_risk,
            calibrated_probability=f_probs,
            confidence=f_conf,
            uncertainty=f_unc,
            quality=f_qual,
            availability=True,
            reliability=FROZEN_FOOT_RELIABILITY,
            model_version="foot_efficientnet_b3_v1.0",
        )

        # Clinical Modality (2-class classification)
        c_err = float(rng.normal(0.0, 0.140))
        c_risk = round(float(np.clip(y_star + c_err, 0.0, 1.0)), 6)
        c_probs = (round(1.0 - c_risk, 6), round(c_risk, 6))
        c_conf = round(float(max(c_probs)), 6)
        c_unc = round(float(np.clip(abs(c_err) * 1.4 + rng.uniform(0.01, 0.12), 0.001, 0.60)), 6)
        c_qual = round(float(rng.uniform(0.85, 1.00)), 4)

        clinical_rec = ModalityRecord(
            sample_id=f"SAMP_{packet_id}_CLI",
            modality="clinical",
            risk=c_risk,
            calibrated_probability=c_probs,
            confidence=c_conf,
            uncertainty=c_unc,
            quality=c_qual,
            availability=True,
            reliability=FROZEN_CLINICAL_RELIABILITY,
            model_version="catboost_hpo_130hosp_v1",
        )

        pkt = ControlledDecisionPacket(
            packet_id=packet_id,
            retina=retina_rec,
            foot=foot_rec,
            clinical=clinical_rec,
            seed=int(seed),
        )
        packets.append(pkt)

    return packets

