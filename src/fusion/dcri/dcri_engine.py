"""
FusionMedAI - Phase C11.6: DCRI Orchestration Engine & Sensitivity Analyzer
Orchestrates ACARA-U routing weights, risk projections, and uncertainty discounting to produce DCRI results.
"""

from typing import Dict, Any, List, Optional, Tuple, Sequence
import math
import numpy as np

from src.fusion.baselines.decision_packet import ControlledDecisionPacket, ModalityRecord
from src.fusion.router.acarau_router import ACARAUv2Router
from src.fusion.router.router_result import RouterResult
from src.fusion.dcri.dcri_result import DCRIResult, DCRIContractError
from src.fusion.dcri.aggregation import (
    compute_weighted_risk_contributions,
    compute_r_fusion,
)
from src.fusion.dcri.uncertainty_penalty import (
    compute_uncertainty_burden,
    compute_uncertainty_penalty_contributions,
    compute_uncertainty_penalty,
)


class DCRIEngine:
    """
    DCRI Dynamic Risk Aggregation & Uncertainty Discounting Engine.
    
    Equations:
        K_i = w_i * r_i
        R_fusion = sum_{i in A} K_i
        U_sum = sum_{i in A} U_i
        U_mean = (1 / |A|) sum_{i in A} U_i
        P_U(delta) = delta * U_sum
        DCRI_delta = R_fusion - P_U(delta)
    """

    DEFAULT_DELTA_GRID: Tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0)
    # NOTE: DEFAULT_DELTA = 0.20 represents a provisional convenience operating point
    # inherited from historical project notes. It is NOT a selected or locked hyperparameter.
    # Formal empirical selection of delta belongs strictly to Phase C11.12 (Delta Selection Protocol).
    DEFAULT_DELTA: float = 0.20

    def __init__(
        self,
        router: Optional[ACARAUv2Router] = None,
        delta_grid: Optional[Sequence[float]] = None,
    ):
        """
        Initializes DCRI Engine with dynamic router and sensitivity grid.
        
        Args:
            router: ACARA-U dynamic router instance.
            delta_grid: Pre-specified grid of uncertainty penalty multipliers.
        """
        self.router = router if router is not None else ACARAUv2Router()
        self.delta_grid = tuple(float(d) for d in (delta_grid if delta_grid is not None else self.DEFAULT_DELTA_GRID))

    def evaluate_packet(
        self,
        packet: ControlledDecisionPacket,
        delta: float = DEFAULT_DELTA,
        router_result: Optional[RouterResult] = None,
    ) -> DCRIResult:
        """
        Evaluates a single decision packet at a specific uncertainty penalty delta.
        
        Args:
            packet: ControlledDecisionPacket instance.
            delta: Uncertainty scaling coefficient (delta >= 0.0). Provisional default 0.20 (not locked).
            router_result: Optional pre-computed RouterResult. If None, computed live.
            
        Returns:
            DCRIResult: Immutable evaluation contract.
        """
        active_modalities = packet.available_modalities
        num_active = len(active_modalities)

        # 1. Zero-Modality Safe Rejection
        if num_active == 0:
            return DCRIResult.no_modality_available(
                packet_id=packet.packet_id,
                delta=float(delta),
            )

        # 2. Modality Weights from Router
        if router_result is None:
            router_input = packet.to_router_input()
            router_result = self.router.route(router_input)

        weights = {m: float(router_result.weights.get(m, 0.0)) for m in ["retina", "foot", "clinical"]}
        risks = {m: float(packet.records[m].risk) for m in ["retina", "foot", "clinical"]}
        uncertainties = {m: float(packet.records[m].uncertainty) for m in ["retina", "foot", "clinical"]}

        # 3. Weighted Risk Contributions & Aggregation: K_i = w_i * r_i, R_fusion = sum K_i
        weighted_contributions = compute_weighted_risk_contributions(
            weights=weights,
            risks=risks,
            active_modalities=active_modalities,
        )
        r_fusion = compute_r_fusion(
            weighted_contributions=weighted_contributions,
            active_modalities=active_modalities,
        )

        # 4. Uncertainty Burden & Discount: U_sum, U_mean, P_U(delta) = delta * U_sum
        u_sum, u_mean = compute_uncertainty_burden(
            uncertainties=uncertainties,
            active_modalities=active_modalities,
        )
        penalty_contributions = compute_uncertainty_penalty_contributions(
            uncertainties=uncertainties,
            delta=float(delta),
            active_modalities=active_modalities,
        )
        penalty = compute_uncertainty_penalty(
            u_sum=u_sum,
            delta=float(delta),
        )

        # 5. Derived Decision-Critical Risk Index: DCRI = R_fusion - P_U(delta)
        # Note: Bounded in [-delta * |A|, 1.0], NOT clamped to [0, 1].
        dcri = float(r_fusion - penalty)

        return DCRIResult(
            packet_id=packet.packet_id,
            r_fusion=r_fusion,
            u_sum=u_sum,
            u_mean=u_mean,
            uncertainty_penalty=penalty,
            dcri=dcri,
            delta=float(delta),
            modality_weights=weights,
            modality_risks=risks,
            weighted_risk_contributions=weighted_contributions,
            modality_uncertainties=uncertainties,
            uncertainty_penalty_contributions=penalty_contributions,
            active_modalities=active_modalities,
            num_active=num_active,
            status="SUCCESS",
        )

    def evaluate_packet_across_grid(
        self,
        packet: ControlledDecisionPacket,
        grid: Optional[Sequence[float]] = None,
    ) -> Dict[float, DCRIResult]:
        """
        Evaluates a single packet across the complete delta sensitivity grid.
        
        Args:
            packet: ControlledDecisionPacket.
            grid: Sequence of delta values (defaults to self.delta_grid).
            
        Returns:
            Dict mapping delta to corresponding DCRIResult.
        """
        target_grid = tuple(float(d) for d in (grid if grid is not None else self.delta_grid))
        
        # Route once to reuse router weights
        if packet.num_available == 0:
            router_res = None
        else:
            router_input = packet.to_router_input()
            router_res = self.router.route(router_input)

        return {
            d: self.evaluate_packet(packet, delta=d, router_result=router_res)
            for d in target_grid
        }

    def evaluate_cohort(
        self,
        packets: List[ControlledDecisionPacket],
        delta_grid: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates a full cohort of decision packets across the delta grid,
        computing distribution statistics, sensitivity metrics, and modality contribution breakdowns.
        """
        target_grid = tuple(float(d) for d in (delta_grid if delta_grid is not None else self.delta_grid))
        n_packets = len(packets)

        # Pre-route packets once
        grid_results: Dict[float, List[DCRIResult]] = {d: [] for d in target_grid}

        for p in packets:
            router_res = self.router.route(p.to_router_input()) if p.num_available > 0 else None
            for d in target_grid:
                res = self.evaluate_packet(p, delta=d, router_result=router_res)
                grid_results[d].append(res)

        # Compute common statistics on R_fusion and Uncertainty Burden (independent of delta)
        ref_results = grid_results[target_grid[0]] if target_grid else []
        success_ref = [r for r in ref_results if r.status == "SUCCESS"]

        r_fusions = [r.r_fusion for r in success_ref]
        u_sums = [r.u_sum for r in success_ref]
        u_means = [r.u_mean for r in success_ref]

        weights_r = [r.modality_weights["retina"] for r in success_ref]
        weights_f = [r.modality_weights["foot"] for r in success_ref]
        weights_c = [r.modality_weights["clinical"] for r in success_ref]

        k_r = [r.weighted_risk_contributions["retina"] for r in success_ref]
        k_f = [r.weighted_risk_contributions["foot"] for r in success_ref]
        k_c = [r.weighted_risk_contributions["clinical"] for r in success_ref]

        delta_statistics: Dict[str, Any] = {}
        for d in target_grid:
            d_res_list = [r for r in grid_results[d] if r.status == "SUCCESS"]
            dcris = [r.dcri for r in d_res_list]
            penalties = [r.uncertainty_penalty for r in d_res_list]
            neg_count = sum(1 for x in dcris if x < 0.0)

            delta_statistics[f"delta_{d}"] = {
                "delta": d,
                "dcri": self._calc_stats(dcris),
                "penalty": self._calc_stats(penalties),
                "negative_count": neg_count,
                "negative_rate": round(float(neg_count / max(len(dcris), 1)), 6),
                "mean_delta_dcri": round(float(np.mean(dcris) - np.mean(r_fusions)), 6) if dcris else 0.0,
            }

        return {
            "num_packets": n_packets,
            "num_success": len(success_ref),
            "delta_grid": list(target_grid),
            "r_fusion_statistics": self._calc_stats(r_fusions),
            "uncertainty_sum_statistics": self._calc_stats(u_sums),
            "uncertainty_mean_statistics": self._calc_stats(u_means),
            "modality_weights_statistics": {
                "retina": self._calc_stats(weights_r),
                "foot": self._calc_stats(weights_f),
                "clinical": self._calc_stats(weights_c),
            },
            "weighted_risk_contributions_statistics": {
                "retina": self._calc_stats(k_r),
                "foot": self._calc_stats(k_f),
                "clinical": self._calc_stats(k_c),
            },
            "delta_sensitivity": delta_statistics,
            "sample_results": [
                {
                    "packet_id": packets[i].packet_id,
                    "evaluations": {
                        f"delta_{d}": grid_results[d][i].to_dict() for d in target_grid
                    }
                }
                for i in range(min(5, n_packets))
            ],
        }

    @staticmethod
    def _calc_stats(values: List[float]) -> Dict[str, float]:
        """Computes mean, median, std, min, max, and IQR."""
        if not values:
            return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "iqr": 0.0}
        arr = np.asarray(values, dtype=np.float64)
        q75, q25 = np.percentile(arr, [75, 25])
        iqr = float(q75 - q25)
        return {
            "mean": round(float(np.mean(arr)), 6),
            "median": round(float(np.median(arr)), 6),
            "std": round(float(np.std(arr)), 6),
            "min": round(float(np.min(arr)), 6),
            "max": round(float(np.max(arr)), 6),
            "iqr": round(float(iqr), 6),
        }
