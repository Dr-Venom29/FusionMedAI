"""
FusionMedAI - Phase C11.12: Sensitivity Metrics & Behavioral Response Calculators
Implements decision metrics, conflict indicators, sensitivity slopes, normalized sensitivity,
and exact logit derivative verification.
"""

from typing import Dict, List, Tuple, Any, Optional
import math
import numpy as np

from src.fusion.baselines.decision_packet import ControlledDecisionPacket
from src.fusion.router.acarau_router import ACARAUv2Router
from src.fusion.router.coefficients import RouterCoefficients
from src.fusion.router.router_result import RouterResult


def compute_packet_metrics(
    packet: ControlledDecisionPacket,
    router_result: RouterResult,
    delta: float = 0.20,
) -> Dict[str, Any]:
    """
    Computes all decision-level metrics and conflict indicators for a single packet under a given router output.
    
    Args:
        packet: ControlledDecisionPacket containing constituent modality records.
        router_result: Output from ACARAUv2Router.
        delta: Provisional DCRI uncertainty penalty coefficient (default 0.20).
        
    Returns:
        Dictionary of packet metrics.
    """
    active_modalities = router_result.active_modalities
    num_active = len(active_modalities)
    weights = router_result.weights

    if num_active == 0:
        return {
            "packet_id": packet.packet_id,
            "status": "NO_MODALITY_AVAILABLE",
            "weights": {"retina": 0.0, "foot": 0.0, "clinical": 0.0},
            "r_fusion": 0.0,
            "entropy": 0.0,
            "dominant_modality": None,
            "dcri": 0.0,
            "uncertainty_sum": 0.0,
            "delta_max": 0.0,
            "delta_mean": 0.0,
            "sigma_w": 0.0,
            "active_modalities": [],
            "num_active": 0,
        }

    # Extract risk projections and uncertainties
    records = packet.records
    risks = {m: float(records[m].risk) for m in ["retina", "foot", "clinical"]}
    uncertainties = {m: float(records[m].uncertainty) for m in ["retina", "foot", "clinical"]}

    # Fused Risk: R_fusion = sum_{i in A} w_i * r_i
    r_fusion = sum(weights[m] * risks[m] for m in active_modalities)
    if r_fusion < -1e-7 or r_fusion > 1.0 + 1e-7:
        raise ValueError(f"Packet {packet.packet_id}: Fused risk R_fusion={r_fusion:.6f} out of bounds [0, 1]")
    r_fusion = max(0.0, min(1.0, float(r_fusion)))

    # DCRI: R_fusion - delta * sum_{i in A} U_i
    active_u_sum = sum(uncertainties[m] for m in active_modalities)
    dcri = float(r_fusion - delta * active_u_sum)

    # Entropy: H(w) = -sum_{i in A, w_i > 0} w_i * ln(w_i)
    entropy = router_result.routing_entropy
    dominant_modality = router_result.dominant_modality

    # Pairwise Conflict Metrics
    if num_active >= 2:
        diffs = []
        for i in range(len(active_modalities)):
            for j in range(i + 1, len(active_modalities)):
                m1, m2 = active_modalities[i], active_modalities[j]
                diffs.append(abs(risks[m1] - risks[m2]))
        delta_max = float(max(diffs))
        delta_mean = float(np.mean(diffs))
    else:
        delta_max = 0.0
        delta_mean = 0.0

    # Weighted Consensus Standard Deviation: sigma_w = sqrt(sum w_i * (r_i - R_fusion)^2)
    if num_active >= 2:
        var_w = sum(weights[m] * ((risks[m] - r_fusion) ** 2) for m in active_modalities)
        sigma_w = float(math.sqrt(max(0.0, var_w)))
    else:
        sigma_w = 0.0

    return {
        "packet_id": packet.packet_id,
        "status": "SUCCESS",
        "weights": {m: round(float(weights.get(m, 0.0)), 6) for m in ["retina", "foot", "clinical"]},
        "r_fusion": round(float(r_fusion), 6),
        "entropy": round(float(entropy), 6),
        "dominant_modality": dominant_modality,
        "dcri": round(float(dcri), 6),
        "uncertainty_sum": round(float(active_u_sum), 6),
        "delta_max": round(float(delta_max), 6),
        "delta_mean": round(float(delta_mean), 6),
        "sigma_w": round(float(sigma_w), 6),
        "active_modalities": list(active_modalities),
        "num_active": num_active,
    }


def aggregate_cohort_metrics(packet_metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes summary distribution statistics across the evaluated cohort of packets.
    """
    valid_metrics = [m for m in packet_metrics_list if m["status"] == "SUCCESS"]
    n_valid = len(valid_metrics)

    if n_valid == 0:
        return {"n_packets": 0, "status": "EMPTY"}

    w_retina = [m["weights"]["retina"] for m in valid_metrics]
    w_foot = [m["weights"]["foot"] for m in valid_metrics]
    w_clinical = [m["weights"]["clinical"] for m in valid_metrics]
    entropies = [m["entropy"] for m in valid_metrics]
    r_fusions = [m["r_fusion"] for m in valid_metrics]
    dcris = [m["dcri"] for m in valid_metrics]
    delta_maxs = [m["delta_max"] for m in valid_metrics]
    delta_means = [m["delta_mean"] for m in valid_metrics]
    sigma_ws = [m["sigma_w"] for m in valid_metrics]

    # Dominance counts
    dom_counts = {"retina": 0, "foot": 0, "clinical": 0}
    for m in valid_metrics:
        dom = m.get("dominant_modality")
        if dom in dom_counts:
            dom_counts[dom] += 1

    return {
        "n_packets": n_valid,
        "weights": {
            "retina": {
                "mean": round(float(np.mean(w_retina)), 6),
                "std": round(float(np.std(w_retina)), 6),
                "min": round(float(np.min(w_retina)), 6),
                "max": round(float(np.max(w_retina)), 6),
            },
            "foot": {
                "mean": round(float(np.mean(w_foot)), 6),
                "std": round(float(np.std(w_foot)), 6),
                "min": round(float(np.min(w_foot)), 6),
                "max": round(float(np.max(w_foot)), 6),
            },
            "clinical": {
                "mean": round(float(np.mean(w_clinical)), 6),
                "std": round(float(np.std(w_clinical)), 6),
                "min": round(float(np.min(w_clinical)), 6),
                "max": round(float(np.max(w_clinical)), 6),
            },
        },
        "routing_entropy": {
            "mean": round(float(np.mean(entropies)), 6),
            "std": round(float(np.std(entropies)), 6),
            "min": round(float(np.min(entropies)), 6),
            "max": round(float(np.max(entropies)), 6),
        },
        "r_fusion": {
            "mean": round(float(np.mean(r_fusions)), 6),
            "std": round(float(np.std(r_fusions)), 6),
            "min": round(float(np.min(r_fusions)), 6),
            "max": round(float(np.max(r_fusions)), 6),
        },
        "dcri": {
            "mean": round(float(np.mean(dcris)), 6),
            "std": round(float(np.std(dcris)), 6),
            "min": round(float(np.min(dcris)), 6),
            "max": round(float(np.max(dcris)), 6),
        },
        "conflict": {
            "delta_max_mean": round(float(np.mean(delta_maxs)), 6),
            "delta_mean_mean": round(float(np.mean(delta_means)), 6),
            "sigma_w_mean": round(float(np.mean(sigma_ws)), 6),
        },
        "dominance_rate": {
            "retina": round(float(dom_counts["retina"] / n_valid), 6),
            "foot": round(float(dom_counts["foot"] / n_valid), 6),
            "clinical": round(float(dom_counts["clinical"] / n_valid), 6),
        },
    }


def compute_sensitivity_slopes(
    param_values: List[float],
    metric_values: List[float],
    param_ref: float,
    metric_ref: float,
) -> Dict[str, float]:
    """
    Computes empirical sensitivity slope S_theta and normalized sensitivity slope S_theta^norm.
    
    Linear Slope:
        S_theta = (M(theta_max) - M(theta_min)) / (theta_max - theta_min)
        
    Normalized Slope:
        S_theta^norm = ( (M(theta_max) - M(theta_min)) / M_ref ) / ( (theta_max - theta_min) / theta_ref )
    """
    min_idx = int(np.argmin(param_values))
    max_idx = int(np.argmax(param_values))

    delta_theta = param_values[max_idx] - param_values[min_idx]
    delta_m = metric_values[max_idx] - metric_values[min_idx]

    if abs(delta_theta) < 1e-12:
        return {"linear_slope": 0.0, "normalized_slope": 0.0}

    linear_slope = delta_m / delta_theta

    if abs(metric_ref) > 1e-12 and abs(param_ref) > 1e-12:
        normalized_slope = (delta_m / metric_ref) / (delta_theta / param_ref)
    else:
        normalized_slope = 0.0

    return {
        "linear_slope": round(float(linear_slope), 6),
        "normalized_slope": round(float(normalized_slope), 6),
        "delta_param": round(float(delta_theta), 6),
        "delta_metric": round(float(delta_m), 6),
    }


def verify_logit_derivatives(
    packet: ControlledDecisionPacket,
    coeff_1: RouterCoefficients,
    coeff_2: RouterCoefficients,
) -> Dict[str, float]:
    """
    Verifies the mathematically exact derivative relationship for relative logit differences:
        Delta(z_i - z_j) = Delta_alpha * (C_i - C_j)
                         + Delta_beta  * (R_i - R_j)
                         - Delta_gamma * (U_i - U_j)
                         + Delta_eta   * (Q_i - Q_j)
                         
    Returns maximum discrepancy across all modality pairs.
    """
    router_1 = ACARAUv2Router(coefficients=coeff_1)
    router_2 = ACARAUv2Router(coefficients=coeff_2)

    router_input = packet.to_router_input()
    raw_logits_1, _ = router_1.compute_logits(router_input)
    raw_logits_2, _ = router_2.compute_logits(router_input)

    d_alpha = float(coeff_2.alpha - coeff_1.alpha)
    d_beta = float(coeff_2.beta - coeff_1.beta)
    d_gamma = float(coeff_2.gamma - coeff_1.gamma)
    d_eta = float(coeff_2.eta - coeff_1.eta)

    ch = router_input.channels
    mods = ["retina", "foot", "clinical"]
    max_error = 0.0

    for i in range(len(mods)):
        for j in range(i + 1, len(mods)):
            m_i, m_j = mods[i], mods[j]

            # Empirical logit difference change
            diff_1 = raw_logits_1[m_i] - raw_logits_1[m_j]
            diff_2 = raw_logits_2[m_i] - raw_logits_2[m_j]
            empirical_delta = diff_2 - diff_1

            # Theoretical expected change
            expected_delta = (
                d_alpha * (ch[m_i].confidence - ch[m_j].confidence)
                + d_beta * (ch[m_i].reliability - ch[m_j].reliability)
                - d_gamma * (ch[m_i].uncertainty - ch[m_j].uncertainty)
                + d_eta * (ch[m_i].quality - ch[m_j].quality)
            )

            err = abs(empirical_delta - expected_delta)
            if err > max_error:
                max_error = err

    return {"max_error": float(max_error)}


def compute_aggregate_sensitivity(
    sweep_slopes: Dict[str, Dict[str, Dict[str, float]]]
) -> Dict[str, Dict[str, float]]:
    """
    Computes unified aggregate sensitivity metrics for each coefficient across all 3 modality weight dimensions:
        S_theta^agg = sqrt( S_theta(w_R)^2 + S_theta(w_F)^2 + S_theta(w_C)^2 )
        S_theta^agg,norm = sqrt( S_theta^norm(w_R)^2 + S_theta^norm(w_F)^2 + S_theta^norm(w_C)^2 )
        
    Provides a mathematically principled, multi-dimensional metric for ranking parameter responsiveness
    without arbitrary single-channel cherry-picking.
    """
    agg_scores: Dict[str, Dict[str, float]] = {}

    for param_name, metrics in sweep_slopes.items():
        s_r = metrics["w_retina"]["linear_slope"]
        s_f = metrics["w_foot"]["linear_slope"]
        s_c = metrics["w_clinical"]["linear_slope"]

        sn_r = metrics["w_retina"]["normalized_slope"]
        sn_f = metrics["w_foot"]["normalized_slope"]
        sn_c = metrics["w_clinical"]["normalized_slope"]

        linear_norm = math.sqrt(s_r ** 2 + s_f ** 2 + s_c ** 2)
        normalized_norm = math.sqrt(sn_r ** 2 + sn_f ** 2 + sn_c ** 2)

        agg_scores[param_name] = {
            "linear_aggregate_norm": round(float(linear_norm), 6),
            "normalized_aggregate_norm": round(float(normalized_norm), 6),
            "linear_slopes": {"retina": s_r, "foot": s_f, "clinical": s_c},
            "normalized_slopes": {"retina": sn_r, "foot": sn_f, "clinical": sn_c},
        }

    return agg_scores
