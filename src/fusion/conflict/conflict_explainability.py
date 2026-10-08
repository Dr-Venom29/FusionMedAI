"""
src/fusion/conflict/conflict_explainability.py
Phase C11.7: Cross-Modality Conflict & Discordance Analysis

Generates deterministic, structured diagnostic explanations for cross-modality conflict
without non-deterministic language models.
"""

from typing import Any, Dict
from .conflict_result import ConflictResult


def generate_conflict_summary(result: ConflictResult) -> Dict[str, Any]:
    """
    Constructs a deterministic, structured diagnostic explanation of conflict state.
    """
    if result.num_active == 0:
        return {
            "status": "NO_INPUT_AVAILABLE",
            "summary": "No modalities were available for evaluation.",
            "conflict_metrics": {
                "max_disagreement": None,
                "mean_disagreement": None,
                "weighted_std": 0.0,
            },
            "dominant_pair": None,
        }

    if result.num_active == 1:
        mod = result.active_modalities[0]
        return {
            "status": "SINGLE_MODALITY_EVALUATION",
            "summary": f"Single modality '{mod}' available; cross-modality conflict cannot be computed.",
            "conflict_metrics": {
                "max_disagreement": None,
                "mean_disagreement": None,
                "weighted_std": 0.0,
            },
            "dominant_pair": None,
        }

    # Multi-modality diagnostic summary
    pair_str = f"{result.dominant_conflict_pair[0]} ↔ {result.dominant_conflict_pair[1]}" if result.dominant_conflict_pair else "none"
    
    severity_labels = {
        "LOW": "Low Decision Conflict (High Inter-Modality Consensus)",
        "MODERATE": "Moderate Decision Conflict",
        "HIGH": "High Decision Conflict Alert",
    }
    status_label = severity_labels.get(result.conflict_severity, result.conflict_severity)

    explanation = (
        f"{status_label}: Maximum pairwise risk divergence Delta_max = {result.max_disagreement:.4f} "
        f"(mean = {result.mean_disagreement:.4f}, weighted sigma = {result.weighted_std:.4f}), "
        f"primarily between {pair_str}. "
        f"ACARA-U assigns highest decision authority to '{result.dominant_modality}' "
        f"(weight = {result.max_weight:.4f}, routing entropy = {result.weight_entropy:.4f}). "
        f"Aggregated Fused Risk R_fusion = {result.r_fusion:.4f}, DCRI = {result.dcri:.4f} "
        f"(cumulative uncertainty burden U_sum = {result.uncertainty_sum:.4f})."
    )

    pair_details = []
    for rec in result.pairwise_records:
        pair_details.append({
            "pair": f"{rec.modality_a}-{rec.modality_b}",
            "risk_diff": rec.absolute_difference,
            "signed_diff": rec.signed_difference,
            "higher_risk_channel": rec.higher_risk_modality,
            "dominant_authority_channel": rec.dominant_authority_modality,
            "authority_weights": {rec.modality_a: rec.weight_a, rec.modality_b: rec.weight_b},
            "uncertainties": {rec.modality_a: rec.uncertainty_a, rec.modality_b: rec.uncertainty_b},
            "reliabilities": {rec.modality_a: rec.reliability_a, rec.modality_b: rec.reliability_b},
            "reliability_authority_aligned": rec.reliability_authority_aligned,
        })

    return {
        "status": result.conflict_severity,
        "summary": explanation,
        "conflict_metrics": {
            "max_disagreement": result.max_disagreement,
            "mean_disagreement": result.mean_disagreement,
            "weighted_variance": result.weighted_variance,
            "weighted_std": result.weighted_std,
            "routing_entropy": result.weight_entropy,
        },
        "dominant_pair": result.dominant_conflict_pair,
        "dominant_modality": result.dominant_modality,
        "max_authority_weight": result.max_weight,
        "r_fusion": result.r_fusion,
        "dcri": result.dcri,
        "pairwise_breakdown": pair_details,
    }
