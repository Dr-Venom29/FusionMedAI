"""
FusionMedAI - Historical Controlled Research Risk Aggregation Module
Computes weighted modality risk contributions K_i = w_i * r_i and aggregated index R_fusion = sum_i K_i.

WARNING - SCIENTIFIC SEMANTICS & CLINICAL ISOLATION:
This calculation represents a weighted average of heterogeneous ordinal and binary diagnostic projections
(diabetic retinopathy severity, Wagner foot ulcer grade, 30-day hospital readmission).
It is retained strictly for historical controlled research benchmark reproduction (Phases C11.1–C11.17).
It DOES NOT measure unified patient risk, a common clinical probability, or validated triage acuity.
Active clinical workflows must consume the structured EvidenceReport instead.
"""

from typing import Dict, List, Tuple
import math
import numpy as np


def compute_weighted_risk_contributions(
    weights: Dict[str, float],
    risks: Dict[str, float],
    active_modalities: List[str],
) -> Dict[str, float]:
    """
    Computes individual weighted modality risk contributions K_i = w_i * r_i.
    
    Args:
        weights: Modality routing weights w_i.
        risks: Modality scalar risk projections r_i in [0.0, 1.0].
        active_modalities: List of available modality identifiers.
        
    Returns:
        Dict mapping modality name to weighted contribution K_i.
    """
    contributions: Dict[str, float] = {}
    active_set = set(active_modalities)

    for mod in ["retina", "foot", "clinical"]:
        if mod in active_set:
            w = float(weights.get(mod, 0.0))
            r = float(risks.get(mod, 0.0))
            if math.isnan(w) or math.isnan(r) or math.isinf(w) or math.isinf(r):
                raise ValueError(f"Non-finite weight or risk encountered for modality '{mod}': w={w}, r={r}")
            if w < -1e-7 or w > 1.0 + 1e-7:
                raise ValueError(f"Modality '{mod}' weight w={w:.6f} out of bounds [0.0, 1.0]")
            if r < -1e-7 or r > 1.0 + 1e-7:
                raise ValueError(f"Modality '{mod}' risk r={r:.6f} out of bounds [0.0, 1.0]")
            
            k = w * r
            contributions[mod] = float(k)
        else:
            contributions[mod] = 0.0

    # Defensive check: sum of active weights must equal 1.0
    if active_modalities:
        active_weight_sum = sum(float(weights.get(m, 0.0)) for m in active_modalities)
        if abs(active_weight_sum - 1.0) > 1e-5:
            raise ValueError(f"Active weights sum ({active_weight_sum:.6f}) does not equal 1.0.")

    return contributions


def compute_r_fusion(
    weighted_contributions: Dict[str, float],
    active_modalities: List[str],
) -> float:
    """
    Computes aggregated decision-level risk index R_fusion = sum_{i in A} K_i.
    
    Args:
        weighted_contributions: Decomposition K_i for all modalities.
        active_modalities: List of available modalities.
        
    Returns:
        Aggregated continuous risk index R_fusion in [0.0, 1.0].
    """
    if not active_modalities:
        return 0.0

    r_fusion = sum(float(weighted_contributions.get(m, 0.0)) for m in active_modalities)
    
    # Numerical tolerance clip for precision rounding
    if -1e-7 <= r_fusion < 0.0:
        r_fusion = 0.0
    elif 1.0 < r_fusion <= 1.0 + 1e-7:
        r_fusion = 1.0

    if r_fusion < 0.0 or r_fusion > 1.0:
        raise ValueError(f"Aggregated R_fusion = {r_fusion:.6f} out of bounds [0.0, 1.0].")

    return float(r_fusion)
