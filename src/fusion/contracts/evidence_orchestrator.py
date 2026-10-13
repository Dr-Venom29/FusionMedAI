"""
FusionMedAI - Heterogeneous Clinical Evidence Orchestration Engine
Coordinates constituent diagnostic predictions, routing authority weights,
and structured evidence reports without conflating heterogeneous clinical targets.
"""

from typing import Dict, Any, List, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from src.fusion.router.acarau_router import ACARAUv2Router

from src.fusion.contracts.evidence_report import (
    ModalityPrediction,
    RoutingResult,
    EvidenceReport,
    ResearchFusionIndex,
    CrossTargetAggregationError,
    SemanticContractError,
    RETINA_TARGET,
    FOOT_TARGET,
    CLINICAL_TARGET,
)


class EvidenceOrchestrator:
    """
    Heterogeneous Evidence Orchestrator.
    Produces structured patient evidence reports combining modality predictions
    and routing authority without calculating a unified patient-level clinical probability.
    """

    def __init__(
        self,
        router: Optional["ACARAUv2Router"] = None,
    ):
        if router is not None:
            self.router = router
        else:
            from src.fusion.router.acarau_router import ACARAUv2Router
            self.router = ACARAUv2Router()

    def orchestrate_evidence(
        self,
        patient_id: str,
        predictions: List[ModalityPrediction],
        routing_purpose: str = "computational_attention_allocation",
    ) -> EvidenceReport:
        """
        Orchestrates evidence across available modality predictions.

        Args:
            patient_id: Patient or encounter identifier.
            predictions: List of ModalityPrediction objects.
            routing_purpose: Purpose for routing authority ('computational_attention_allocation', etc.).

        Returns:
            EvidenceReport: Structured clinical evidence report.
        """
        if not isinstance(predictions, (list, tuple)):
            raise SemanticContractError(
                f"predictions must be a list or tuple of ModalityPrediction instances, got {type(predictions).__name__}."
            )

        for idx, p in enumerate(predictions):
            if not isinstance(p, ModalityPrediction):
                raise SemanticContractError(
                    f"All predictions must be instances of ModalityPrediction; item at index {idx} has type {type(p).__name__}."
                )

        # Validate that no two predictions share the same modality
        modality_names = [p.modality for p in predictions]
        if len(modality_names) != len(set(modality_names)):
            raise SemanticContractError(f"Duplicate modality predictions encountered: {modality_names}")

        # Map predictions by modality
        pred_map = {p.modality: p for p in predictions}
        active_modalities = [p.modality for p in predictions if p.available]

        if not active_modalities:
            # Safe fail-closed empty report
            empty_routing = RoutingResult(
                weights={"retina": 0.0, "foot": 0.0, "clinical": 0.0},
                active_modalities=[],
                routing_policy_version=self.router.router_version,
                purpose=routing_purpose,
                entropy=0.0,
            )
            return EvidenceReport(
                patient_id=patient_id,
                predictions=predictions,
                routing=empty_routing,
                status="NO_MODALITY_AVAILABLE",
                common_target_prediction=None,
            )

        # Build router inputs from predictions
        r_pred = pred_map.get("retina")
        f_pred = pred_map.get("foot")
        c_pred = pred_map.get("clinical")

        from src.fusion.router.router_input import RouterInput, ModalityChannelInput, FROZEN_RELIABILITY_MAP

        def _make_channel(mod: str, pred: Optional[ModalityPrediction]) -> ModalityChannelInput:
            expected_rel = FROZEN_RELIABILITY_MAP[mod]
            if pred is None or not pred.available:
                return ModalityChannelInput(
                    modality=mod,
                    confidence=0.0,
                    reliability=expected_rel,
                    uncertainty=1.0,
                    quality=0.0,
                    availability=False,
                )

            # Reconcile reliability: prediction reliability must match frozen validation prior
            if abs(pred.reliability - expected_rel) > 1e-4:
                raise SemanticContractError(
                    f"Reliability mismatch for modality '{mod}': prediction has {pred.reliability:.6f}, "
                    f"but frozen prior requires {expected_rel:.6f}."
                )

            # Extract modality-specific canonical confidence feature:
            probs = pred.get_calibrated_probability()
            if mod == "retina":
                # Margin between top two predicted stages: p_(5) - p_(4)
                s = sorted(probs) if len(probs) >= 2 else [0.0, 1.0]
                conf = float(np.clip(s[-1] - s[-2], 0.0, 1.0))
            elif mod == "foot":
                # Maximum class probability: max_k p_k
                conf = float(np.clip(max(probs), 0.0, 1.0)) if probs else 0.0
            elif mod == "clinical":
                # Margin distance from decision boundary: |p1 - 0.5| * 2
                p1 = probs[1] if len(probs) == 2 else 0.5
                conf = float(np.clip(abs(p1 - 0.5) * 2.0, 0.0, 1.0))
            else:
                conf = 0.0

            return ModalityChannelInput(
                modality=mod,
                confidence=conf,
                reliability=expected_rel,
                uncertainty=pred.get_uncertainty_value(),
                quality=pred.quality,
                availability=True,
            )

        r_input = RouterInput(
            retina=_make_channel("retina", r_pred),
            foot=_make_channel("foot", f_pred),
            clinical=_make_channel("clinical", c_pred),
        )

        router_res = self.router.route(r_input)
        routing_res = router_res.as_routing_result(purpose=routing_purpose)

        # Active evidence reports strictly exclude common_target_prediction
        return EvidenceReport(
            patient_id=patient_id,
            predictions=predictions,
            routing=routing_res,
            status="SUCCESS",
            common_target_prediction=None,
        )

    def compute_historical_research_index(
        self,
        predictions: List[ModalityPrediction],
        delta: float = 0.10,
    ) -> ResearchFusionIndex:
        """
        Dedicated historical research benchmark function.
        STRICTLY DECOUPLED FROM ACTIVE CLINICAL EVIDENCE REPORTS.
        Computes weighted projection index R_fusion and DCRI for reproduction of C11.1–C11.17.
        """
        import math
        if not isinstance(delta, (int, float)) or isinstance(delta, bool) or not math.isfinite(delta) or delta < 0.0:
            raise SemanticContractError(f"delta penalty parameter must be a non-negative finite float, got {delta!r}")

        if not isinstance(predictions, (list, tuple)):
            raise SemanticContractError("predictions must be a list or tuple of ModalityPrediction instances.")

        for p in predictions:
            if not isinstance(p, ModalityPrediction):
                raise SemanticContractError(f"Historical calculation requires ModalityPrediction instances, got {type(p).__name__}")

        mod_names = [p.modality for p in predictions]
        if len(mod_names) != len(set(mod_names)):
            raise SemanticContractError(f"Duplicate modality predictions encountered in historical calculation: {mod_names}")

        pred_map = {p.modality: p for p in predictions}
        active_modalities = [p.modality for p in predictions if p.available]

        if not active_modalities:
            return ResearchFusionIndex(
                index_value=0.0,
                dcri_value=0.0,
                weights={"retina": 0.0, "foot": 0.0, "clinical": 0.0},
                uncertainty_penalty=0.0,
                is_clinical_probability=False,
                clinical_action_authorized=False,
            )

        # Route using modality-specific confidence features matching C11
        from src.fusion.router.router_input import RouterInput, ModalityChannelInput, FROZEN_RELIABILITY_MAP
        def _get_ch(mod: str) -> ModalityChannelInput:
            p = pred_map.get(mod)
            rel = FROZEN_RELIABILITY_MAP[mod]
            if p is None or not p.available:
                return ModalityChannelInput(modality=mod, confidence=0.0, reliability=rel, uncertainty=1.0, quality=0.0, availability=False)
            probs = p.get_calibrated_probability()
            if mod == "retina":
                s = sorted(probs) if len(probs) >= 2 else [0.0, 1.0]
                conf = float(np.clip(s[-1] - s[-2], 0.0, 1.0))
            elif mod == "foot":
                conf = float(np.clip(max(probs), 0.0, 1.0)) if probs else 0.0
            elif mod == "clinical":
                p1 = probs[1] if len(probs) == 2 else 0.5
                conf = float(np.clip(abs(p1 - 0.5) * 2.0, 0.0, 1.0))
            else:
                conf = 0.0
            return ModalityChannelInput(modality=mod, confidence=conf, reliability=rel, uncertainty=p.get_uncertainty_value(), quality=p.quality, availability=True)

        r_in = RouterInput(retina=_get_ch("retina"), foot=_get_ch("foot"), clinical=_get_ch("clinical"))
        r_res = self.router.route(r_in)
        weights = r_res.weights

        scalars: Dict[str, float] = {"retina": 0.0, "foot": 0.0, "clinical": 0.0}
        for m in active_modalities:
            raw_scalar = pred_map[m].legacy_projected_scalar
            if (
                raw_scalar is None
                or not isinstance(raw_scalar, (int, float))
                or isinstance(raw_scalar, bool)
                or not math.isfinite(raw_scalar)
                or raw_scalar < 0.0
                or raw_scalar > 1.0
            ):
                raise SemanticContractError(
                    f"Active modality '{m}' in historical index calculation requires a finite "
                    f"legacy_projected_scalar in [0.0, 1.0], got {raw_scalar!r}."
                )
            scalars[m] = float(raw_scalar)

        r_fusion = sum(weights.get(m, 0.0) * scalars.get(m, 0.0) for m in active_modalities)
        u_sum = sum(pred_map[m].get_uncertainty_value() for m in active_modalities)
        p_u = delta * u_sum
        dcri = r_fusion - p_u

        return ResearchFusionIndex(
            index_value=round(float(r_fusion), 6),
            dcri_value=round(float(dcri), 6),
            weights={m: round(float(w), 6) for m, w in weights.items()},
            uncertainty_penalty=round(float(p_u), 6),
            is_clinical_probability=False,
            clinical_action_authorized=False,
        )

    @staticmethod
    def calculate_unified_risk_probability(*args, **kwargs):
        """
        Guards against unauthorized cross-target mathematical conflation.
        """
        raise CrossTargetAggregationError(
            "CRITICAL SCIENTIFIC ERROR: Cannot calculate a unified patient risk probability from "
            "heterogeneous diagnostic targets (diabetic retinopathy severity, Wagner foot ulcer grade, "
            "and 30-day hospital readmission). A common-target prediction requires a separately trained "
            "and clinically validated model on genuinely paired patient data."
        )
