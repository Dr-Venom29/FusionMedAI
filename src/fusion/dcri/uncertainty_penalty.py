"""
FusionMedAI - Phase C11.6: Uncertainty Penalty & Burden Engine
Calculates cumulative uncertainty burden U_sum, mean uncertainty U_mean, and uncertainty penalty P_U(delta) = delta * U_sum.
Strictly defensive with full parameter bounds checking.
"""

from typing import Dict, List, Tuple
import math


def compute_uncertainty_burden(
    uncertainties: Dict[str, float],
    active_modalities: List[str],
) -> Tuple[float, float]:
    """
    Computes cumulative uncertainty burden U_sum and average uncertainty U_mean.
    
    Args:
        uncertainties: Modality predictive uncertainties U_i in [0.0, 1.0].
        active_modalities: List of available modality identifiers.
        
    Returns:
        Tuple of (U_sum, U_mean).
    """
    if not active_modalities:
        return 0.0, 0.0

    active_u: List[float] = []
    for mod in active_modalities:
        u = float(uncertainties.get(mod, 0.0))
        if math.isnan(u) or math.isinf(u):
            raise ValueError(f"Non-finite uncertainty encountered for modality '{mod}': {u}")
        if u < -1e-7 or u > 1.0 + 1e-7:
            raise ValueError(f"Uncertainty for modality '{mod}' ({u:.6f}) out of bounds [0.0, 1.0].")
        active_u.append(max(0.0, min(1.0, u)))

    u_sum = sum(active_u)
    u_mean = u_sum / len(active_modalities)

    return float(u_sum), float(u_mean)


def compute_uncertainty_penalty_contributions(
    uncertainties: Dict[str, float],
    delta: float,
    active_modalities: List[str],
) -> Dict[str, float]:
    """
    Computes per-modality uncertainty penalty contributions p_i = delta * U_i.
    
    Args:
        uncertainties: Modality predictive uncertainties U_i in [0.0, 1.0].
        delta: Uncertainty penalty scaling coefficient (delta >= 0.0).
        active_modalities: List of available modalities.
        
    Returns:
        Dict mapping modality name to penalty contribution p_i.
    """
    if not isinstance(delta, (int, float)) or math.isnan(delta) or math.isinf(delta):
        raise ValueError(f"delta must be a finite float, got {delta}")
    if delta < 0.0:
        raise ValueError(f"delta must be non-negative, got {delta}")

    contributions: Dict[str, float] = {}
    active_set = set(active_modalities)
    for mod in ["retina", "foot", "clinical"]:
        if mod in active_set:
            u = float(uncertainties.get(mod, 0.0))
            if math.isnan(u) or math.isinf(u):
                raise ValueError(f"Non-finite uncertainty for modality '{mod}': {u}")
            if u < -1e-7 or u > 1.0 + 1e-7:
                raise ValueError(f"Uncertainty for modality '{mod}' ({u:.6f}) out of bounds [0.0, 1.0].")
            contributions[mod] = float(delta * max(0.0, min(1.0, u)))
        else:
            contributions[mod] = 0.0

    return contributions


def compute_uncertainty_penalty(
    u_sum: float,
    delta: float,
) -> float:
    """
    Computes total uncertainty penalty P_U(delta) = delta * U_sum.
    
    Args:
        u_sum: Cumulative uncertainty sum sum_{i in A} U_i.
        delta: Uncertainty penalty scaling coefficient (delta >= 0.0).
        
    Returns:
        Total uncertainty penalty P_U(delta).
    """
    if not isinstance(delta, (int, float)) or math.isnan(delta) or math.isinf(delta):
        raise ValueError(f"delta must be a finite float, got {delta}")
    if delta < 0.0:
        raise ValueError(f"delta must be non-negative, got {delta}")
    if not isinstance(u_sum, (int, float)) or math.isnan(u_sum) or math.isinf(u_sum):
        raise ValueError(f"Non-finite u_sum: {u_sum}")
    if u_sum < -1e-7:
        raise ValueError(f"u_sum cannot be negative: {u_sum}")

    penalty = delta * float(u_sum)
    return float(penalty)
