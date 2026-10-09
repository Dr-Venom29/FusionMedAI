"""
FusionMedAI - Phase C11.13: Regime-Stratified Delta Evaluator
Evaluates candidate delta parameters across all 7 active modality availability regimes
(R, F, C, RF, RC, FC, RFC) and verifies fail-closed EMPTY regime handling.
"""

from typing import Dict, List, Tuple, Any
import numpy as np

from src.fusion.baselines.decision_packet import ControlledDecisionPacket
from src.fusion.router.acarau_router import ACARAUv2Router
from src.fusion.router.coefficients import RouterCoefficients
from src.fusion.router.router_input import RouterInput, ModalityChannelInput
from .selection_config import (
    ACTIVE_REGIMES,
    ALL_REGIMES,
    ALPHA_REF,
    BETA_REF,
    GAMMA_REF,
    ETA_REF,
    MODALITIES,
)
from .candidate_grid import DeltaCandidateItem


REGIME_MODALITY_MAP: Dict[str, List[str]] = {
    "R": ["retina"],
    "F": ["foot"],
    "C": ["clinical"],
    "RF": ["retina", "foot"],
    "RC": ["retina", "clinical"],
    "FC": ["foot", "clinical"],
    "RFC": ["retina", "foot", "clinical"],
}


class RegimeEvaluator:
    """
    Computes regime-stratified metrics across all 7 active regimes and EMPTY.
    """

    def __init__(self):
        self.coeff = RouterCoefficients(
            alpha=ALPHA_REF,
            beta=BETA_REF,
            gamma=GAMMA_REF,
            eta=ETA_REF,
        )
        self.router = ACARAUv2Router(coefficients=self.coeff)

    def precompute_regime_base_states(
        self,
        cohort: List[ControlledDecisionPacket],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Precomputes base states (w_i, r_i, U_i, R_fusion, U_sum) for all 7 active regimes across cohort.
        """
        regime_base_states: Dict[str, List[Dict[str, Any]]] = {reg: [] for reg in ACTIVE_REGIMES}

        for reg in ACTIVE_REGIMES:
            active_mods = REGIME_MODALITY_MAP[reg]
            for pkt in cohort:
                # Build masked router input
                ch_map = {}
                for m in MODALITIES:
                    rec = getattr(pkt, m)
                    is_active = (m in active_mods)
                    ch_map[m] = ModalityChannelInput(
                        modality=m,
                        confidence=float(rec.confidence),
                        reliability=float(rec.reliability),
                        uncertainty=float(rec.uncertainty),
                        quality=float(rec.quality) if is_active else 0.0,
                        availability=is_active,
                    )
                r_in = RouterInput(retina=ch_map["retina"], foot=ch_map["foot"], clinical=ch_map["clinical"])
                r_out = self.router.route(r_in)

                weights = {m: r_out.weights.get(m, 0.0) for m in active_mods}
                risks = {}
                uncertainties = {}
                r_fusion = 0.0
                u_sum = 0.0

                for m in active_mods:
                    rec = getattr(pkt, m)
                    r_i = float(rec.risk)
                    u_i = float(rec.uncertainty)
                    w_i = float(weights[m])
                    risks[m] = r_i
                    uncertainties[m] = u_i
                    r_fusion += w_i * r_i
                    u_sum += u_i

                regime_base_states[reg].append({
                    "packet_id": pkt.packet_id,
                    "regime": reg,
                    "active_mods": active_mods,
                    "weights": weights,
                    "risks": risks,
                    "uncertainties": uncertainties,
                    "r_fusion": float(r_fusion),
                    "u_sum": float(u_sum),
                })

        return regime_base_states

    def evaluate_regimes(
        self,
        candidate_evals: List[Dict[str, Any]],
        cohort: List[ControlledDecisionPacket],
    ) -> Dict[str, Any]:
        """
        Computes per-regime breakdowns for every candidate delta across all 7 regimes.
        """
        regime_base_states = self.precompute_regime_base_states(cohort)
        regime_counts = {reg: len(cohort) for reg in ACTIVE_REGIMES}
        regime_counts["EMPTY"] = len(cohort)

        by_candidate: Dict[str, Any] = {}

        for cand_eval in candidate_evals:
            cid = cand_eval["candidate_id"]
            delta = cand_eval["delta"]
            cand_regime_dict: Dict[str, Any] = {}

            for reg in ACTIVE_REGIMES:
                states = regime_base_states[reg]
                dcri_list = []
                pen_list = []
                u_sum_list = []
                r_f_list = []

                for s in states:
                    r_f = s["r_fusion"]
                    u_s = s["u_sum"]
                    pen = delta * u_s
                    dcri = r_f - pen
                    dcri_list.append(dcri)
                    pen_list.append(pen)
                    u_sum_list.append(u_s)
                    r_f_list.append(r_f)

                dcri_arr = np.array(dcri_list, dtype=np.float64)
                pen_arr = np.array(pen_list, dtype=np.float64)
                u_sum_arr = np.array(u_sum_list, dtype=np.float64)

                cardinality = len(REGIME_MODALITY_MAP[reg])
                n_neg = int(np.sum(dcri_arr < 0.0))
                neg_rate = float(n_neg / len(dcri_arr))

                cand_regime_dict[reg] = {
                    "regime": reg,
                    "n_packets": len(states),
                    "cardinality": cardinality,
                    "mean_dcri": float(np.mean(dcri_arr)),
                    "median_dcri": float(np.median(dcri_arr)),
                    "std_dcri": float(np.std(dcri_arr, ddof=1)),
                    "min_dcri": float(np.min(dcri_arr)),
                    "max_dcri": float(np.max(dcri_arr)),
                    "negative_count": n_neg,
                    "negative_rate": neg_rate,
                    "mean_penalty": float(np.mean(pen_arr)),
                    "mean_u_sum": float(np.mean(u_sum_arr)),
                    "mean_r_fusion": float(np.mean(r_f_list)),
                    "fail_closed_safe": True,
                }

            # Exercise actual empty fail-closed routing path through ACARA-U router
            from src.fusion.reliability.global_reliability import (
                FROZEN_RETINA_RELIABILITY,
                FROZEN_FOOT_RELIABILITY,
                FROZEN_CLINICAL_RELIABILITY,
            )
            empty_in = RouterInput(
                retina=ModalityChannelInput(modality="retina", confidence=0.0, reliability=FROZEN_RETINA_RELIABILITY, uncertainty=1.0, quality=0.0, availability=False),
                foot=ModalityChannelInput(modality="foot", confidence=0.0, reliability=FROZEN_FOOT_RELIABILITY, uncertainty=1.0, quality=0.0, availability=False),
                clinical=ModalityChannelInput(modality="clinical", confidence=0.0, reliability=FROZEN_CLINICAL_RELIABILITY, uncertainty=1.0, quality=0.0, availability=False),
            )
            empty_out = self.router.route(empty_in)
            is_fail_closed = (
                empty_out.status == "NO_MODALITY_AVAILABLE"
                and empty_out.num_active == 0
                and len(empty_out.active_modalities) == 0
                and all(w == 0.0 for w in empty_out.weights.values())
                and empty_out.routing_entropy == 0.0
            )

            cand_regime_dict["EMPTY"] = {
                "regime": "EMPTY",
                "n_packets": len(cohort),
                "cardinality": 0,
                "router_status": empty_out.status,
                "mean_dcri": None,
                "median_dcri": None,
                "std_dcri": None,
                "negative_rate": 0.0,
                "mean_penalty": 0.0,
                "mean_u_sum": 0.0,
                "fail_closed_safe": is_fail_closed,
            }

            by_candidate[cid] = {
                "candidate_id": cid,
                "delta": delta,
                "regimes": cand_regime_dict,
            }

        # Cross-candidate summary table for key metrics
        regime_matrix_mean_dcri: Dict[str, Dict[str, float]] = {reg: {} for reg in ACTIVE_REGIMES}
        regime_matrix_negative_rate: Dict[str, Dict[str, float]] = {reg: {} for reg in ACTIVE_REGIMES}
        regime_matrix_mean_penalty: Dict[str, Dict[str, float]] = {reg: {} for reg in ACTIVE_REGIMES}

        for cand_eval in candidate_evals:
            cid = cand_eval["candidate_id"]
            for reg in ACTIVE_REGIMES:
                reg_data = by_candidate[cid]["regimes"][reg]
                regime_matrix_mean_dcri[reg][cid] = reg_data["mean_dcri"]
                regime_matrix_negative_rate[reg][cid] = reg_data["negative_rate"]
                regime_matrix_mean_penalty[reg][cid] = reg_data["mean_penalty"]

        return {
            "regime_counts": regime_counts,
            "by_candidate": by_candidate,
            "regime_matrices": {
                "mean_dcri": regime_matrix_mean_dcri,
                "negative_rate": regime_matrix_negative_rate,
                "mean_penalty": regime_matrix_mean_penalty,
            },
        }
