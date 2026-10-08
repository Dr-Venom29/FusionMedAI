"""
src/fusion/conflict/pairwise_disagreement.py
Phase C11.7: Cross-Modality Conflict & Discordance Analysis

Computes pairwise risk divergence, directionality, and authority attribution
between all active modality pairs.
"""

from typing import Dict, List, Tuple
from itertools import combinations
from .conflict_result import PairwiseRecord


def compute_pairwise_record(
    modality_a: str,
    modality_b: str,
    risk_a: float,
    risk_b: float,
    weight_a: float,
    weight_b: float,
    uncertainty_a: float,
    uncertainty_b: float,
    reliability_a: float,
    reliability_b: float,
) -> PairwiseRecord:
    """
    Computes directional and absolute disagreement between two modalities.
    """
    signed_diff = float(risk_a - risk_b)
    abs_diff = float(abs(signed_diff))

    # Determine higher / lower risk
    if abs(signed_diff) < 1e-9:
        higher_risk = "tied"
        lower_risk = "tied"
    elif signed_diff > 0:
        higher_risk = modality_a
        lower_risk = modality_b
    else:
        higher_risk = modality_b
        lower_risk = modality_a

    # Determine dominant authority
    if abs(weight_a - weight_b) < 1e-9:
        dominant_authority = "tied"
    elif weight_a > weight_b:
        dominant_authority = modality_a
    else:
        dominant_authority = modality_b

    # Reliability-authority alignment:
    # True if the modality with higher global reliability receives >= router authority
    if abs(reliability_a - reliability_b) < 1e-9:
        rel_auth_aligned = True
    elif reliability_a > reliability_b:
        rel_auth_aligned = (weight_a >= weight_b - 1e-9)
    else:
        rel_auth_aligned = (weight_b >= weight_a - 1e-9)

    return PairwiseRecord(
        modality_a=modality_a,
        modality_b=modality_b,
        risk_a=float(risk_a),
        risk_b=float(risk_b),
        signed_difference=signed_diff,
        absolute_difference=abs_diff,
        higher_risk_modality=higher_risk,
        lower_risk_modality=lower_risk,
        weight_a=float(weight_a),
        weight_b=float(weight_b),
        dominant_authority_modality=dominant_authority,
        uncertainty_a=float(uncertainty_a),
        uncertainty_b=float(uncertainty_b),
        reliability_a=float(reliability_a),
        reliability_b=float(reliability_b),
        reliability_authority_aligned=rel_auth_aligned,
    )


def compute_all_pairwise_records(
    active_modalities: Tuple[str, ...],
    risks: Dict[str, float],
    weights: Dict[str, float],
    uncertainties: Dict[str, float],
    reliabilities: Dict[str, float],
) -> Tuple[PairwiseRecord, ...]:
    """
    Computes pairwise records for all unordered pairs of active modalities.
    Canonical ordering: combinations of active_modalities preserving order.
    """
    if len(active_modalities) < 2:
        return ()

    records: List[PairwiseRecord] = []
    for mod_a, mod_b in combinations(active_modalities, 2):
        rec = compute_pairwise_record(
            modality_a=mod_a,
            modality_b=mod_b,
            risk_a=risks[mod_a],
            risk_b=risks[mod_b],
            weight_a=weights[mod_a],
            weight_b=weights[mod_b],
            uncertainty_a=uncertainties[mod_a],
            uncertainty_b=uncertainties[mod_b],
            reliability_a=reliabilities[mod_a],
            reliability_b=reliabilities[mod_b],
        )
        records.append(rec)

    return tuple(records)
