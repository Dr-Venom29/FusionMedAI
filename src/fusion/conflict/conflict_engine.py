"""
src/fusion/conflict/conflict_engine.py
Phase C11.7: Cross-Modality Conflict & Discordance Analysis

Orchestrates ACARA-U routing, DCRI calculation, and cross-modality conflict analysis.
Evaluates pairwise divergence, weighted consensus dispersion, and operational severity
without modifying upstream router weights or DCRI values.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import copy

from src.fusion.baselines.decision_packet import ControlledDecisionPacket, ModalityRecord
from src.fusion.router.acarau_router import ACARAUv2Router
from src.fusion.dcri.dcri_engine import DCRIEngine
from .conflict_result import ConflictResult, PairwiseRecord
from .pairwise_disagreement import compute_all_pairwise_records
from .conflict_metrics import (
    compute_max_disagreement,
    compute_mean_disagreement,
    compute_weighted_dispersion,
    compute_weight_entropy,
)
from .conflict_classifier import classify_conflict_severity


class ConflictEngine:
    """
    Cross-Modality Conflict & Discordance Analysis Engine.
    
    Equations:
        X_jk = |r_j - r_k|
        Delta_max = max_{j < k} |r_j - r_k|
        Delta_mean = (1 / P) sum_{j < k} |r_j - r_k|
        V_w = sum_{i in A} w_i * (r_i - R_fusion)^2
        sigma_w = sqrt(V_w)
    """

    def __init__(
        self,
        router: Optional[ACARAUv2Router] = None,
        dcri_engine: Optional[DCRIEngine] = None,
    ):
        """
        Initializes Conflict Engine with optional router and DCRI engine.
        """
        self.router = router or ACARAUv2Router()
        self.dcri_engine = dcri_engine or DCRIEngine(router=self.router)

    def evaluate_packet(
        self,
        packet: ControlledDecisionPacket,
        delta: float = 0.20,
    ) -> ConflictResult:
        """
        Evaluates cross-modality conflict for a single ControlledDecisionPacket.
        """
        # Execute upstream router and DCRI engine
        dcri_res = self.dcri_engine.evaluate_packet(packet=packet, delta=delta)
        
        return self.evaluate_from_packet_and_dcri(packet=packet, dcri_result=dcri_res)

    def evaluate_from_packet_and_dcri(
        self,
        packet: ControlledDecisionPacket,
        dcri_result: Any,
    ) -> ConflictResult:
        """
        Evaluates conflict using an already-evaluated DCRIResult, ensuring strict upstream immutability.
        """
        active_modalities = tuple(dcri_result.active_modalities)
        num_active = dcri_result.num_active

        risks: Dict[str, float] = {}
        uncertainties: Dict[str, float] = {}
        reliabilities: Dict[str, float] = {}
        weights: Dict[str, float] = dict(dcri_result.modality_weights)

        for mod in active_modalities:
            if mod in packet.records:
                rec = packet.records[mod]
                risks[mod] = float(rec.risk)
                uncertainties[mod] = float(rec.uncertainty)
                reliabilities[mod] = float(rec.reliability)

        # Handle zero and single modality cases
        if num_active == 0:
            return ConflictResult(
                packet_id=packet.packet_id,
                active_modalities=(),
                num_active=0,
                conflict_available=False,
                max_disagreement=None,
                mean_disagreement=None,
                weighted_variance=None,
                weighted_std=None,
                pairwise_records=(),
                dominant_conflict_pair=None,
                conflict_severity="NO_MODALITY_AVAILABLE",
                weight_entropy=0.0,
                max_weight=0.0,
                dominant_modality="none",
                r_fusion=0.0,
                dcri=0.0,
                uncertainty_sum=0.0,
                uncertainty_mean=0.0,
            )

        if num_active == 1:
            dominant_mod = active_modalities[0]
            return ConflictResult(
                packet_id=packet.packet_id,
                active_modalities=active_modalities,
                num_active=1,
                conflict_available=False,
                max_disagreement=None,
                mean_disagreement=None,
                weighted_variance=0.0,
                weighted_std=0.0,
                pairwise_records=(),
                dominant_conflict_pair=None,
                conflict_severity="NOT_APPLICABLE",
                weight_entropy=0.0,
                max_weight=1.0,
                dominant_modality=dominant_mod,
                r_fusion=float(dcri_result.r_fusion),
                dcri=float(dcri_result.dcri),
                uncertainty_sum=float(dcri_result.u_sum if hasattr(dcri_result, "u_sum") else dcri_result.uncertainty_burden_sum),
                uncertainty_mean=float(dcri_result.u_mean if hasattr(dcri_result, "u_mean") else dcri_result.uncertainty_burden_mean),
            )

        # Multi-modality case (num_active >= 2)
        pairwise_recs = compute_all_pairwise_records(
            active_modalities=active_modalities,
            risks=risks,
            weights=weights,
            uncertainties=uncertainties,
            reliabilities=reliabilities,
        )

        max_disag, dominant_pair = compute_max_disagreement(pairwise_recs)
        mean_disag = compute_mean_disagreement(pairwise_recs)
        weighted_var, weighted_std = compute_weighted_dispersion(
            active_modalities=active_modalities,
            risks=risks,
            weights=weights,
            r_fusion=dcri_result.r_fusion,
        )
        entropy = compute_weight_entropy(active_modalities=active_modalities, weights=weights)

        # Find dominant authority modality
        max_w = -1.0
        dom_mod = "tied"
        for mod in active_modalities:
            w = weights.get(mod, 0.0)
            if w > max_w:
                max_w = w
                dom_mod = mod

        severity = classify_conflict_severity(num_active=num_active, max_disagreement=max_disag)

        return ConflictResult(
            packet_id=packet.packet_id,
            active_modalities=active_modalities,
            num_active=num_active,
            conflict_available=True,
            max_disagreement=max_disag,
            mean_disagreement=mean_disag,
            weighted_variance=weighted_var,
            weighted_std=weighted_std,
            pairwise_records=pairwise_recs,
            dominant_conflict_pair=dominant_pair,
            conflict_severity=severity,
            weight_entropy=entropy,
            max_weight=float(max_w),
            dominant_modality=dom_mod,
            r_fusion=float(dcri_result.r_fusion),
            dcri=float(dcri_result.dcri),
            uncertainty_sum=float(dcri_result.u_sum if hasattr(dcri_result, "u_sum") else dcri_result.uncertainty_burden_sum),
            uncertainty_mean=float(dcri_result.u_mean if hasattr(dcri_result, "u_mean") else dcri_result.uncertainty_burden_mean),
        )

    def evaluate_cohort(
        self,
        packets: Sequence[ControlledDecisionPacket],
        delta: float = 0.20,
    ) -> List[ConflictResult]:
        """
        Evaluates full cohort of decision packets.
        """
        return [self.evaluate_packet(packet=p, delta=delta) for p in packets]

    def evaluate_modality_configurations(
        self,
        packet: ControlledDecisionPacket,
        delta: float = 0.20,
    ) -> Dict[str, ConflictResult]:
        """
        Evaluates a packet across all 7 modality availability regimes plus zero-modality.
        Regimes: retina_only, foot_only, clinical_only, retina_foot, retina_clinical, foot_clinical, tri_modal, zero_modality.
        """
        regimes: Dict[str, Tuple[str, ...]] = {
            "retina_only": ("retina",),
            "foot_only": ("foot",),
            "clinical_only": ("clinical",),
            "retina_foot": ("retina", "foot"),
            "retina_clinical": ("retina", "clinical"),
            "foot_clinical": ("foot", "clinical"),
            "tri_modal": ("retina", "foot", "clinical"),
            "zero_modality": (),
        }

        results: Dict[str, ConflictResult] = {}
        for regime_name, active_mods in regimes.items():
            records = {}
            for m in ["retina", "foot", "clinical"]:
                old_rec = packet.records[m]
                is_avail = (m in active_mods)
                records[m] = ModalityRecord(
                    sample_id=old_rec.sample_id,
                    modality=old_rec.modality,
                    risk=old_rec.risk,
                    calibrated_probability=old_rec.calibrated_probability,
                    confidence=old_rec.confidence if is_avail else 0.0,
                    uncertainty=old_rec.uncertainty if is_avail else 0.0,
                    quality=old_rec.quality if is_avail else 0.0,
                    availability=bool(is_avail),
                    reliability=old_rec.reliability,
                    model_version=old_rec.model_version,
                )
            masked_packet = ControlledDecisionPacket(
                packet_id=f"{packet.packet_id}_{regime_name}",
                retina=records["retina"],
                foot=records["foot"],
                clinical=records["clinical"],
                seed=packet.seed,
            )
            res = self.evaluate_packet(packet=masked_packet, delta=delta)
            results[regime_name] = res

        return results

    def perturb_modality_risk(
        self,
        packet: ControlledDecisionPacket,
        target_modality: str,
        new_risk: float,
        delta: float = 0.20,
    ) -> ConflictResult:
        """
        Creates a perturbed copy of a decision packet with a new risk value for target_modality,
        and evaluates the resulting conflict state.
        """
        if target_modality not in {"retina", "foot", "clinical"}:
            raise ValueError(f"Unknown target_modality '{target_modality}'")
        if not (0.0 <= new_risk <= 1.0):
            raise ValueError(f"new_risk={new_risk} must be in [0.0, 1.0]")

        if target_modality not in packet.records:
            raise ValueError(f"Modality '{target_modality}' not found in packet {packet.packet_id}")

        old_rec = packet.records[target_modality]

        new_rec = ModalityRecord(
            sample_id=str(old_rec.sample_id),
            modality=str(old_rec.modality),
            risk=float(new_risk),
            calibrated_probability=old_rec.calibrated_probability,
            confidence=float(old_rec.confidence),
            uncertainty=float(old_rec.uncertainty),
            quality=float(old_rec.quality),
            availability=bool(old_rec.availability),
            reliability=float(old_rec.reliability),
            model_version=str(old_rec.model_version),
        )

        perturbed_dict = {
            "retina": packet.retina,
            "foot": packet.foot,
            "clinical": packet.clinical,
        }
        perturbed_dict[target_modality] = new_rec

        perturbed_packet = ControlledDecisionPacket(
            packet_id=f"{packet.packet_id}_perturbed_{target_modality}_{new_risk:.2f}",
            retina=perturbed_dict["retina"],
            foot=perturbed_dict["foot"],
            clinical=perturbed_dict["clinical"],
            seed=packet.seed,
        )

        return self.evaluate_packet(packet=perturbed_packet, delta=delta)
