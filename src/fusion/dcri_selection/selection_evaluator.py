"""
FusionMedAI - Phase C11.13: DCRI Selection Evaluator
Evaluates candidate delta parameters across the frozen cohort (N=500, seed 115)
under reference router Theta_0 = (1.0, 1.5, 1.0, 0.5).
"""

from typing import Dict, List, Tuple, Any, Optional
import math
import numpy as np
from scipy import stats

from src.fusion.baselines.decision_packet import ControlledDecisionPacket
from src.fusion.router.acarau_router import ACARAUv2Router
from src.fusion.router.coefficients import RouterCoefficients
from src.fusion.router.router_input import RouterInput, ModalityChannelInput
from .selection_config import (
    ALPHA_REF,
    BETA_REF,
    GAMMA_REF,
    ETA_REF,
    MODALITIES,
    MAX_ACCEPTABLE_FLOAT_TOLERANCE,
)
from .candidate_grid import DeltaCandidateItem


def build_router_input_from_packet(packet: ControlledDecisionPacket) -> RouterInput:
    """Builds RouterInput from a ControlledDecisionPacket."""
    return packet.to_router_input()


class SelectionEvaluator:
    """
    Evaluates candidate delta parameters over the frozen cohort.
    Ensures upstream weights, calibrated risks, and uncertainties are computed once
    and strictly frozen across all candidate delta conditions.
    """

    def __init__(self):
        self.coeff = RouterCoefficients(
            alpha=ALPHA_REF,
            beta=BETA_REF,
            gamma=GAMMA_REF,
            eta=ETA_REF,
        )
        self.router = ACARAUv2Router(coefficients=self.coeff)

    def precompute_packet_base_state(
        self,
        cohort: List[ControlledDecisionPacket],
    ) -> List[Dict[str, Any]]:
        """
        Precomputes base fusion state (w_i, r_i, U_i, R_fusion, U_sum) once per packet.
        
        Returns:
            List of packet base state dictionaries.
        """
        base_states: List[Dict[str, Any]] = []
        
        for packet in cohort:
            router_input = packet.to_router_input()
            router_res = self.router.route(router_input)
            
            # Active modalities
            active_mods = []
            for mod_name in MODALITIES:
                rec = getattr(packet, mod_name, None)
                if rec is not None and rec.availability:
                    active_mods.append(mod_name)
                    
            if not active_mods:
                # EMPTY regime
                base_states.append({
                    "packet_id": packet.packet_id,
                    "regime": "EMPTY",
                    "active_mods": [],
                    "weights": {},
                    "risks": {},
                    "uncertainties": {},
                    "r_fusion": None,
                    "u_sum": 0.0,
                    "is_empty": True,
                })
                continue
                
            weights = {m: router_res.weights.get(m, 0.0) for m in active_mods}
            risks = {}
            uncertainties = {}
            r_fusion = 0.0
            u_sum = 0.0
            
            for m in active_mods:
                rec = getattr(packet, m)
                r_i = rec.risk
                u_i = rec.uncertainty
                w_i = weights[m]
                risks[m] = r_i
                uncertainties[m] = u_i
                r_fusion += w_i * r_i
                u_sum += u_i
                
            # Determine regime string (e.g. 'RFC', 'RF', 'R', etc.)
            regime_order = ["retina", "foot", "clinical"]
            regime_letters = {"retina": "R", "foot": "F", "clinical": "C"}
            regime_str = "".join([regime_letters[m] for m in regime_order if m in active_mods])
            
            base_states.append({
                "packet_id": packet.packet_id,
                "regime": regime_str,
                "active_mods": active_mods,
                "weights": weights,
                "risks": risks,
                "uncertainties": uncertainties,
                "r_fusion": float(r_fusion),
                "u_sum": float(u_sum),
                "is_empty": False,
            })
            
        return base_states

    def evaluate_candidate_delta(
        self,
        candidate: DeltaCandidateItem,
        base_states: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluates a single candidate delta over the precomputed base states.
        
        Args:
            candidate: DeltaCandidateItem.
            base_states: Precomputed base states from precompute_packet_base_state.
            
        Returns:
            Dictionary containing comprehensive behavioral and statistical metrics for this delta.
        """
        delta = candidate.delta
        active_states = [s for s in base_states if not s["is_empty"]]
        
        dcri_values = []
        r_fusion_values = []
        u_sum_values = []
        penalties = []
        packet_evals = []
        
        for s in active_states:
            r_f = s["r_fusion"]
            u_s = s["u_sum"]
            penalty = delta * u_s
            dcri = r_f - penalty
            
            dcri_values.append(dcri)
            r_fusion_values.append(r_f)
            u_sum_values.append(u_s)
            penalties.append(penalty)
            
            packet_evals.append({
                "packet_id": s["packet_id"],
                "regime": s["regime"],
                "r_fusion": r_f,
                "u_sum": u_s,
                "penalty": penalty,
                "dcri": dcri,
                "is_negative": (dcri < 0.0),
            })
            
        dcri_arr = np.array(dcri_values, dtype=np.float64)
        r_f_arr = np.array(r_fusion_values, dtype=np.float64)
        u_s_arr = np.array(u_sum_values, dtype=np.float64)
        pen_arr = np.array(penalties, dtype=np.float64)
        
        n_active = len(dcri_arr)
        n_negative = int(np.sum(dcri_arr < 0.0))
        negative_rate = float(n_negative / n_active) if n_active > 0 else 0.0
        
        # Rank correlations with base risk R_fusion
        if delta == 0.0:
            spearman_rho = 1.0
            spearman_p = 0.0
            kendall_tau = 1.0
            kendall_p = 0.0
        else:
            sp_res = stats.spearmanr(dcri_arr, r_f_arr)
            spearman_rho = float(sp_res.statistic)
            spearman_p = float(sp_res.pvalue)
            kd_res = stats.kendalltau(dcri_arr, r_f_arr)
            kendall_tau = float(kd_res.statistic)
            kendall_p = float(kd_res.pvalue)
            
        mean_dcri = float(np.mean(dcri_arr))
        median_dcri = float(np.median(dcri_arr))
        std_dcri = float(np.std(dcri_arr, ddof=1))
        min_dcri = float(np.min(dcri_arr))
        max_dcri = float(np.max(dcri_arr))
        
        mean_penalty = float(np.mean(pen_arr))
        median_penalty = float(np.median(pen_arr))
        max_penalty = float(np.max(pen_arr))
        mean_rf = float(np.mean(r_f_arr))
        penalty_ratio = float(mean_penalty / mean_rf) if mean_rf > 0 else 0.0
        
        mean_u_sum = float(np.mean(u_s_arr))
        
        # Quantiles
        q_p05 = float(np.percentile(dcri_arr, 5))
        q_q1 = float(np.percentile(dcri_arr, 25))
        q_median = float(np.percentile(dcri_arr, 50))
        q_q3 = float(np.percentile(dcri_arr, 75))
        q_p95 = float(np.percentile(dcri_arr, 95))
        iqr = float(q_q3 - q_q1)
        
        # Penalty exceedance thresholds
        rate_pen_gt_010 = float(np.mean(pen_arr > 0.10))
        rate_pen_gt_020 = float(np.mean(pen_arr > 0.20))
        rate_pen_gt_030 = float(np.mean(pen_arr > 0.30))
        
        # Analytical check: DCRI_delta - R_fusion must equal -delta * u_sum exactly
        max_residual = float(np.max(np.abs((dcri_arr - r_f_arr) - (-pen_arr))))
        
        return {
            "candidate_id": candidate.candidate_id,
            "delta": delta,
            "name": candidate.name,
            "category": candidate.category,
            "description": candidate.description,
            "is_zero": candidate.is_zero,
            "n_active_packets": n_active,
            "mean_dcri": mean_dcri,
            "median_dcri": median_dcri,
            "std_dcri": std_dcri,
            "min_dcri": min_dcri,
            "max_dcri": max_dcri,
            "quantiles": {
                "p05": q_p05,
                "q1": q_q1,
                "median": q_median,
                "q3": q_q3,
                "p95": q_p95,
                "iqr": iqr,
            },
            "negative_count": n_negative,
            "negative_rate": negative_rate,
            "mean_penalty": mean_penalty,
            "median_penalty": median_penalty,
            "max_penalty": max_penalty,
            "mean_u_sum": mean_u_sum,
            "penalty_to_base_ratio": penalty_ratio,
            "penalty_exceedance_rates": {
                "gt_0_10": rate_pen_gt_010,
                "gt_0_20": rate_pen_gt_020,
                "gt_0_30": rate_pen_gt_030,
            },
            "rank_stability": {
                "spearman_rho": spearman_rho,
                "spearman_pvalue": spearman_p,
                "kendall_tau": kendall_tau,
                "kendall_pvalue": kendall_p,
            },
            "max_residual_error": max_residual,
            "packet_evaluations": packet_evals,
        }
