"""
src/fusion/missingness/missingness_engine.py
Phase C11.8: Missing Modality Robustness & Orchestration Engine

Coordinates comprehensive evaluation across:
1. All 8 availability states
2. Full vs subset authority redistribution and DCRI decomposition
3. Sequential information-loss ladders
4. Comparative benchmark against baselines B1–B6
5. Uncertainty and reliability interaction stratifications
6. Masked-value invariance and deterministic stress dropouts
"""

from typing import Dict, Any, List, Tuple, Sequence, Optional
import math
import numpy as np

from src.fusion.baselines.decision_packet import ControlledDecisionPacket
from src.fusion.baselines.fusion_runner import FusionRunner
from src.fusion.dcri.dcri_engine import DCRIEngine
from src.fusion.dcri.dcri_result import DCRIResult
from src.fusion.missingness.availability_mask import (
    ALL_REGIMES,
    NON_EMPTY_REGIMES,
    BIMODAL_REGIMES,
    UNIMODAL_REGIMES,
    MODALITY_NAMES,
    apply_availability_mask,
    verify_availability_invariants,
)
from src.fusion.missingness.dropout_scenarios import (
    get_regime_packet,
    generate_sequential_ladders,
    get_stress_packet,
)
from src.fusion.missingness.robustness_metrics import (
    calc_stats,
    compute_risk_delta,
    compute_dcri_delta,
    compute_authority_redistribution,
    compute_dcri_decomposition,
    compute_routing_entropy,
    compute_max_authority,
    compute_bootstrap_ci,
)
from src.fusion.missingness.missingness_result import (
    RegimeEvaluation,
    RedistributionRecord,
    StressScenarioRecord,
    MaskedInvarianceRecord,
)
from src.fusion.missingness.stress_tests import (
    evaluate_masked_value_invariance,
    evaluate_stress_scenarios,
    evaluate_unavailable_vs_low_quality,
)


class MissingnessEngine:
    """
    Core orchestrator for Missing Modality Robustness (Phase C11.8).
    """

    def __init__(self, dcri_engine: Optional[DCRIEngine] = None):
        self.dcri_engine = dcri_engine if dcri_engine is not None else DCRIEngine()
        self.fusion_runner = FusionRunner()

    def evaluate_packet_regime(
        self,
        packet: ControlledDecisionPacket,
        regime_name: str,
        delta: float = 0.20,
    ) -> RegimeEvaluation:
        """
        Evaluates a single packet under a specified availability regime.
        """
        masked_pkt = get_regime_packet(packet, regime_name)
        active_mods = tuple(masked_pkt.available_modalities)
        num_active = len(active_mods)

        dcri_res: DCRIResult = self.dcri_engine.evaluate_packet(masked_pkt, delta=delta)

        # Invariant check
        is_valid, err = verify_availability_invariants(masked_pkt, dcri_res.modality_weights)
        if not is_valid:
            raise ValueError(f"Availability invariant violation in {regime_name}: {err}")

        entropy = compute_routing_entropy(dcri_res.modality_weights)
        w_max = compute_max_authority(dcri_res.modality_weights)

        # Dominant authority modality
        dom_mod = "none"
        if num_active > 0:
            dom_mod = max(active_mods, key=lambda m: dcri_res.modality_weights.get(m, 0.0))

        return RegimeEvaluation(
            packet_id=packet.packet_id,
            regime_name=regime_name,
            active_modalities=active_mods,
            num_active=num_active,
            weights={m: round(float(dcri_res.modality_weights.get(m, 0.0)), 6) for m in MODALITY_NAMES},
            r_fusion=round(float(dcri_res.r_fusion), 6),
            dcri=round(float(dcri_res.dcri), 6),
            delta=float(delta),
            u_sum=round(float(dcri_res.u_sum), 6),
            u_mean=round(float(dcri_res.u_mean), 6),
            entropy=entropy,
            w_max=w_max,
            dominant_modality=dom_mod,
            status=dcri_res.status,
        )

    def evaluate_packet_all_regimes(
        self,
        packet: ControlledDecisionPacket,
        delta: float = 0.20,
    ) -> Dict[str, RegimeEvaluation]:
        """
        Evaluates a single packet across all 8 availability regimes.
        """
        return {
            reg: self.evaluate_packet_regime(packet, reg, delta=delta)
            for reg in ALL_REGIMES
        }

    def compute_redistribution_record(
        self,
        full_eval: RegimeEvaluation,
        subset_eval: RegimeEvaluation,
        delta: float = 0.20,
    ) -> RedistributionRecord:
        """
        Computes the redistribution record comparing full tri-modal vs subset evaluation.
        """
        removed = tuple(m for m in full_eval.active_modalities if m not in subset_eval.active_modalities)
        delta_w = compute_authority_redistribution(full_eval.weights, subset_eval.weights)
        decomp = compute_dcri_decomposition(
            r_full=full_eval.r_fusion,
            r_subset=subset_eval.r_fusion,
            u_sum_full=full_eval.u_sum,
            u_sum_subset=subset_eval.u_sum,
            delta=delta,
        )

        return RedistributionRecord(
            packet_id=full_eval.packet_id,
            subset_regime=subset_eval.regime_name,
            removed_modalities=removed,
            remaining_modalities=subset_eval.active_modalities,
            delta_w=delta_w,
            delta_r_signed=decomp["delta_r_signed"],
            delta_r_abs=decomp["delta_r_abs"],
            delta_u_sum=decomp["delta_u_sum"],
            delta_penalty=decomp["delta_penalty"],
            delta_dcri_signed=decomp["delta_dcri_signed"],
            delta_dcri_abs=decomp["delta_dcri_abs"],
            entropy_shift=round(subset_eval.entropy - full_eval.entropy, 6),
            w_max_shift=round(subset_eval.w_max - full_eval.w_max, 6),
        )

    def evaluate_cohort_regimes(
        self,
        cohort: Sequence[ControlledDecisionPacket],
        delta: float = 0.20,
    ) -> Dict[str, List[RegimeEvaluation]]:
        """
        Evaluates the full cohort across all 8 availability regimes.
        """
        results: Dict[str, List[RegimeEvaluation]] = {reg: [] for reg in ALL_REGIMES}
        for pkt in cohort:
            for reg in ALL_REGIMES:
                results[reg].append(self.evaluate_packet_regime(pkt, reg, delta=delta))
        return results

    def evaluate_cohort_redistributions(
        self,
        cohort: Sequence[ControlledDecisionPacket],
        delta: float = 0.20,
    ) -> Dict[str, List[RedistributionRecord]]:
        """
        Computes redistribution records comparing full tri-modal against each subset regime.
        """
        subsets = [k for k in NON_EMPTY_REGIMES if k != "tri_modal"]
        redistributions: Dict[str, List[RedistributionRecord]] = {sub: [] for sub in subsets}

        for pkt in cohort:
            full_eval = self.evaluate_packet_regime(pkt, "tri_modal", delta=delta)
            for sub in subsets:
                sub_eval = self.evaluate_packet_regime(pkt, sub, delta=delta)
                rec = self.compute_redistribution_record(full_eval, sub_eval, delta=delta)
                redistributions[sub].append(rec)

        return redistributions

    def evaluate_baselines_comparative(
        self,
        cohort: Sequence[ControlledDecisionPacket],
    ) -> Dict[str, Any]:
        """
        Evaluates Baselines B1–B6 across all availability regimes and calculates
        comparative risk sensitivity under modality dropout.
        """
        baseline_ids = ["B1", "B2", "B3", "B4", "B5", "B6"]
        regimes_to_test = list(NON_EMPTY_REGIMES.keys())

        # baseline -> regime -> list of r_fusion
        evals: Dict[str, Dict[str, List[float]]] = {
            b_id: {reg: [] for reg in regimes_to_test}
            for b_id in baseline_ids
        }

        for pkt in cohort:
            for reg in regimes_to_test:
                masked_pkt = get_regime_packet(pkt, reg)
                for b_id in baseline_ids:
                    res = self.fusion_runner.evaluate_packet(masked_pkt, b_id)
                    evals[b_id][reg].append(res.r_fusion)

        # Compute Delta R for single-modality removals (-R: foot_clinical, -F: retina_clinical, -C: retina_foot)
        dropouts = {
            "missing_retina_FC": "foot_clinical",
            "missing_foot_RC": "retina_clinical",
            "missing_clinical_RF": "retina_foot",
        }

        comparative_results: Dict[str, Any] = {}
        for b_id in baseline_ids:
            comparative_results[b_id] = {}
            for d_name, reg in dropouts.items():
                deltas = [
                    abs(full_r - sub_r)
                    for full_r, sub_r in zip(evals[b_id]["tri_modal"], evals[b_id][reg])
                ]
                comparative_results[b_id][d_name] = {
                    "stats": calc_stats(deltas),
                    "ci_95": compute_bootstrap_ci(deltas),
                }

        return comparative_results

    def evaluate_uncertainty_stratification(
        self,
        cohort: Sequence[ControlledDecisionPacket],
        delta: float = 0.20,
    ) -> Dict[str, Any]:
        """
        Stratifies modality dropout impact by the removed modality's instance-level uncertainty:
        Low uncertainty (U_m < Q25) vs High uncertainty (U_m > Q75).
        """
        strat_results: Dict[str, Any] = {}

        for mod, subset_regime in [("retina", "foot_clinical"), ("foot", "retina_clinical"), ("clinical", "retina_foot")]:
            # Extract uncertainty of modality across cohort
            u_vals = [p.records[mod].uncertainty for p in cohort]
            q25, q75 = float(np.percentile(u_vals, 25)), float(np.percentile(u_vals, 75))

            low_u_indices = [i for i, u in enumerate(u_vals) if u <= q25]
            high_u_indices = [i for i, u in enumerate(u_vals) if u >= q75]

            delta_r_all = []
            delta_dcri_all = []
            delta_w_remaining_all: Dict[str, List[float]] = {}

            for pkt in cohort:
                f_eval = self.evaluate_packet_regime(pkt, "tri_modal", delta=delta)
                s_eval = self.evaluate_packet_regime(pkt, subset_regime, delta=delta)
                rec = self.compute_redistribution_record(f_eval, s_eval, delta=delta)
                delta_r_all.append(rec.delta_r_abs)
                delta_dcri_all.append(rec.delta_dcri_abs)

                for rem_mod, dw in rec.delta_w.items():
                    if rem_mod != mod:
                        if rem_mod not in delta_w_remaining_all:
                            delta_w_remaining_all[rem_mod] = []
                        delta_w_remaining_all[rem_mod].append(dw)

            low_delta_r = [delta_r_all[i] for i in low_u_indices]
            high_delta_r = [delta_r_all[i] for i in high_u_indices]

            low_delta_dcri = [delta_dcri_all[i] for i in low_u_indices]
            high_delta_dcri = [delta_dcri_all[i] for i in high_u_indices]

            strat_results[f"removed_{mod}"] = {
                "uncertainty_thresholds": {"q25": round(q25, 6), "q75": round(q75, 6)},
                "low_uncertainty_stratum": {
                    "count": len(low_u_indices),
                    "delta_r_stats": calc_stats(low_delta_r),
                    "delta_dcri_stats": calc_stats(low_delta_dcri),
                },
                "high_uncertainty_stratum": {
                    "count": len(high_u_indices),
                    "delta_r_stats": calc_stats(high_delta_r),
                    "delta_dcri_stats": calc_stats(high_delta_dcri),
                },
            }

        return strat_results
