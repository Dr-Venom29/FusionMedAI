"""
src/fusion/missingness/explainability.py
Phase C11.8: Deterministic Structured Diagnostic Explainability for Missingness

Generates rule-based, deterministic structured explainability diagnostics
for modality availability transitions, authority redistribution, and DCRI sensitivity.
Strictly zero natural-language LLMs in the scientific loop.
"""

from typing import Dict, Any, List, Optional
from src.fusion.missingness.missingness_result import RegimeEvaluation, RedistributionRecord


def generate_missingness_diagnostic(
    regime_eval: RegimeEvaluation,
    redistribution: Optional[RedistributionRecord] = None,
) -> Dict[str, Any]:
    """
    Generates a deterministic diagnostic dictionary explaining the availability state.
    """
    if regime_eval.num_active == 0:
        return {
            "packet_id": regime_eval.packet_id,
            "regime": regime_eval.regime_name,
            "status": "NO_MODALITY_AVAILABLE",
            "active_modalities": [],
            "missing_modalities": ["retina", "foot", "clinical"],
            "diagnostic_message": "Fail-Closed: All modality channels are unavailable. Fused risk and DCRI rejected safely.",
            "metrics": {
                "r_fusion": 0.0,
                "dcri": 0.0,
                "u_sum": 0.0,
                "routing_entropy": 0.0,
            },
        }

    missing_mods = [m for m in ("retina", "foot", "clinical") if m not in regime_eval.active_modalities]

    diag: Dict[str, Any] = {
        "packet_id": regime_eval.packet_id,
        "regime": regime_eval.regime_name,
        "status": regime_eval.status,
        "num_active": regime_eval.num_active,
        "active_modalities": list(regime_eval.active_modalities),
        "missing_modalities": missing_mods,
        "dominant_authority": regime_eval.dominant_modality,
        "max_authority_weight": regime_eval.w_max,
        "routing_entropy": regime_eval.entropy,
        "fused_risk": regime_eval.r_fusion,
        "dcri": regime_eval.dcri,
        "uncertainty_sum": regime_eval.u_sum,
        "modality_weights": regime_eval.weights,
    }

    if redistribution is not None:
        diag["redistribution_analysis"] = {
            "removed_modalities": list(redistribution.removed_modalities),
            "authority_shifts": redistribution.delta_w,
            "risk_shift_signed": redistribution.delta_r_signed,
            "risk_shift_absolute": redistribution.delta_r_abs,
            "uncertainty_burden_shift": redistribution.delta_u_sum,
            "uncertainty_discount_shift": redistribution.delta_penalty,
            "dcri_shift_signed": redistribution.delta_dcri_signed,
            "dcri_shift_absolute": redistribution.delta_dcri_abs,
            "entropy_shift": redistribution.entropy_shift,
            "max_authority_shift": redistribution.w_max_shift,
        }

    return diag
