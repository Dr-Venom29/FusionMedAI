"""
FusionMedAI - Phase C11.15: Isolated Monotonicity Evaluator
Tests single-input response directions for Confidence (C_i), Uncertainty (U_i),
and Quality (Q_i) through the production ACARAUv2Router.route() path across a
deterministic finite perturbation grid.

Reliability (R_i) monotonicity is evaluated via an isolated mathematical kernel fixture
(z_i = alpha*C + beta*R - gamma*U + eta*Q under beta=1.5 > 0) because historical validation
reliability R_i is an immutable frozen prior in the production pipeline.
Singleton regimes (|A|=1) verify invariant weight w_i = 1.000000 and delta_w = 0.0.
"""

from typing import Dict, List, Tuple, Any, Optional
import math
import numpy as np

from src.fusion.router.acarau_router import ACARAUv2Router
from src.fusion.router.coefficients import RouterCoefficients
from src.fusion.router.router_input import RouterInput, ModalityChannelInput, FROZEN_RELIABILITY_MAP
from .sanity_config import (
    ALPHA_REF,
    BETA_REF,
    GAMMA_REF,
    ETA_REF,
    ROUTER_COEFFS_REF,
    MODALITIES,
    MODALITY_SYMBOLS,
    ACTIVE_REGIMES,
    INTERIOR_GRID,
    BOUNDARY_GRID,
    PERTURBATION_STEPS,
    STRICT_MONOTONIC_EPS,
    FLOAT_TOLERANCE,
)


def create_baseline_channel(
    modality: str,
    confidence: float = 0.5,
    uncertainty: float = 0.2,
    quality: float = 0.8,
    availability: bool = True,
    custom_reliability: Optional[float] = None,
) -> ModalityChannelInput:
    """Creates a valid ModalityChannelInput with default or customized parameters."""
    rel = custom_reliability if custom_reliability is not None else FROZEN_RELIABILITY_MAP[modality]
    q = quality if availability else 0.0
    return ModalityChannelInput(
        modality=modality,
        confidence=confidence,
        reliability=rel,
        uncertainty=uncertainty,
        quality=q,
        availability=availability,
    )


def create_router_input_for_regime(
    regime: str,
    base_confidence: float = 0.5,
    base_uncertainty: float = 0.2,
    base_quality: float = 0.8,
) -> RouterInput:
    """Constructs a RouterInput conforming to the specified availability regime."""
    avail_retina = "R" in regime
    avail_foot = "F" in regime
    avail_clinical = "C" in regime

    return RouterInput(
        retina=create_baseline_channel("retina", base_confidence, base_uncertainty, base_quality, avail_retina),
        foot=create_baseline_channel("foot", base_confidence, base_uncertainty, base_quality, avail_foot),
        clinical=create_baseline_channel("clinical", base_confidence, base_uncertainty, base_quality, avail_clinical),
    )


class IsolatedMonotonicityEvaluator:
    """
    Evaluates isolated single-input monotonicity of the ACARA-U routing engine.
    """

    def __init__(
        self,
        coefficients: Optional[RouterCoefficients] = None,
        router: Optional[ACARAUv2Router] = None,
    ):
        self.coefficients = coefficients or RouterCoefficients(
            alpha=ALPHA_REF,
            beta=BETA_REF,
            gamma=GAMMA_REF,
            eta=ETA_REF,
        )
        self.router = router if router is not None else ACARAUv2Router(coefficients=self.coefficients)

    def evaluate_confidence_monotonicity(self) -> Dict[str, Any]:
        """
        Evaluates d w_i / d C_i:
        Increasing C_i with all other attributes fixed must strictly increase w_i
        for multi-modality active sets (cardinality >= 2), and preserve w_i = 1.0 for single-modality.
        """
        trials: List[Dict[str, Any]] = []
        all_passed = True

        for regime in ACTIVE_REGIMES:
            active_mods = [m for m in MODALITIES if MODALITY_SYMBOLS[m] in regime]
            cardinality = len(active_mods)

            for target_mod in active_mods:
                for base_c in INTERIOR_GRID:
                    for step in PERTURBATION_STEPS:
                        new_c = min(1.0, base_c + step)
                        if new_c <= base_c:
                            continue

                        # Before
                        inp_before = create_router_input_for_regime(regime, base_confidence=base_c)
                        res_before = self.router.route(inp_before)
                        w_before = res_before.weights[target_mod]

                        # After (mutate only target_mod confidence)
                        channels_after = dict(inp_before.channels)
                        target_ch = channels_after[target_mod]
                        channels_after[target_mod] = ModalityChannelInput(
                            modality=target_mod,
                            confidence=new_c,
                            reliability=target_ch.reliability,
                            uncertainty=target_ch.uncertainty,
                            quality=target_ch.quality,
                            availability=target_ch.availability,
                        )
                        inp_after = RouterInput(
                            retina=channels_after["retina"],
                            foot=channels_after["foot"],
                            clinical=channels_after["clinical"],
                        )
                        res_after = self.router.route(inp_after)
                        w_after = res_after.weights[target_mod]

                        delta_w = w_after - w_before
                        if cardinality >= 2:
                            passed = delta_w > STRICT_MONOTONIC_EPS
                        else:
                            passed = abs(w_after - 1.0) < FLOAT_TOLERANCE and abs(delta_w) < FLOAT_TOLERANCE

                        if not passed:
                            all_passed = False

                        trials.append({
                            "regime": regime,
                            "target_modality": target_mod,
                            "cardinality": cardinality,
                            "attribute": "confidence",
                            "val_before": base_c,
                            "val_after": new_c,
                            "weight_before": w_before,
                            "weight_after": w_after,
                            "delta_weight": delta_w,
                            "expected_direction": "INCREASE" if cardinality >= 2 else "INVARIANT_1.0",
                            "passed": passed,
                        })

        return {
            "property": "confidence_monotonicity",
            "total_trials": len(trials),
            "passed_trials": sum(1 for t in trials if t["passed"]),
            "status": "PASS" if all_passed else "FAIL",
            "trials": trials,
        }

    def evaluate_uncertainty_monotonicity(self) -> Dict[str, Any]:
        """
        Evaluates d w_i / d U_i:
        Increasing U_i with all other attributes fixed must strictly decrease w_i
        for multi-modality active sets (cardinality >= 2), and preserve w_i = 1.0 for single-modality.
        """
        trials: List[Dict[str, Any]] = []
        all_passed = True

        for regime in ACTIVE_REGIMES:
            active_mods = [m for m in MODALITIES if MODALITY_SYMBOLS[m] in regime]
            cardinality = len(active_mods)

            for target_mod in active_mods:
                for base_u in INTERIOR_GRID:
                    for step in PERTURBATION_STEPS:
                        new_u = min(1.0, base_u + step)
                        if new_u <= base_u:
                            continue

                        # Before
                        inp_before = create_router_input_for_regime(regime, base_uncertainty=base_u)
                        res_before = self.router.route(inp_before)
                        w_before = res_before.weights[target_mod]

                        # After (mutate only target_mod uncertainty)
                        channels_after = dict(inp_before.channels)
                        target_ch = channels_after[target_mod]
                        channels_after[target_mod] = ModalityChannelInput(
                            modality=target_mod,
                            confidence=target_ch.confidence,
                            reliability=target_ch.reliability,
                            uncertainty=new_u,
                            quality=target_ch.quality,
                            availability=target_ch.availability,
                        )
                        inp_after = RouterInput(
                            retina=channels_after["retina"],
                            foot=channels_after["foot"],
                            clinical=channels_after["clinical"],
                        )
                        res_after = self.router.route(inp_after)
                        w_after = res_after.weights[target_mod]

                        delta_w = w_after - w_before
                        if cardinality >= 2:
                            passed = delta_w < -STRICT_MONOTONIC_EPS
                        else:
                            passed = abs(w_after - 1.0) < FLOAT_TOLERANCE and abs(delta_w) < FLOAT_TOLERANCE

                        if not passed:
                            all_passed = False

                        trials.append({
                            "regime": regime,
                            "target_modality": target_mod,
                            "cardinality": cardinality,
                            "attribute": "uncertainty",
                            "val_before": base_u,
                            "val_after": new_u,
                            "weight_before": w_before,
                            "weight_after": w_after,
                            "delta_weight": delta_w,
                            "expected_direction": "DECREASE" if cardinality >= 2 else "INVARIANT_1.0",
                            "passed": passed,
                        })

        return {
            "property": "uncertainty_monotonicity",
            "total_trials": len(trials),
            "passed_trials": sum(1 for t in trials if t["passed"]),
            "status": "PASS" if all_passed else "FAIL",
            "trials": trials,
        }

    def evaluate_quality_monotonicity(self) -> Dict[str, Any]:
        """
        Evaluates d w_i / d Q_i:
        Increasing Q_i with all other attributes fixed must strictly increase w_i
        for multi-modality active sets (cardinality >= 2), and preserve w_i = 1.0 for single-modality.
        """
        trials: List[Dict[str, Any]] = []
        all_passed = True

        for regime in ACTIVE_REGIMES:
            active_mods = [m for m in MODALITIES if MODALITY_SYMBOLS[m] in regime]
            cardinality = len(active_mods)

            for target_mod in active_mods:
                for base_q in INTERIOR_GRID:
                    for step in PERTURBATION_STEPS:
                        new_q = min(1.0, base_q + step)
                        if new_q <= base_q:
                            continue

                        # Before
                        inp_before = create_router_input_for_regime(regime, base_quality=base_q)
                        res_before = self.router.route(inp_before)
                        w_before = res_before.weights[target_mod]

                        # After (mutate only target_mod quality)
                        channels_after = dict(inp_before.channels)
                        target_ch = channels_after[target_mod]
                        channels_after[target_mod] = ModalityChannelInput(
                            modality=target_mod,
                            confidence=target_ch.confidence,
                            reliability=target_ch.reliability,
                            uncertainty=target_ch.uncertainty,
                            quality=new_q,
                            availability=target_ch.availability,
                        )
                        inp_after = RouterInput(
                            retina=channels_after["retina"],
                            foot=channels_after["foot"],
                            clinical=channels_after["clinical"],
                        )
                        res_after = self.router.route(inp_after)
                        w_after = res_after.weights[target_mod]

                        delta_w = w_after - w_before
                        if cardinality >= 2:
                            passed = delta_w > STRICT_MONOTONIC_EPS
                        else:
                            passed = abs(w_after - 1.0) < FLOAT_TOLERANCE and abs(delta_w) < FLOAT_TOLERANCE

                        if not passed:
                            all_passed = False

                        trials.append({
                            "regime": regime,
                            "target_modality": target_mod,
                            "cardinality": cardinality,
                            "attribute": "quality",
                            "val_before": base_q,
                            "val_after": new_q,
                            "weight_before": w_before,
                            "weight_after": w_after,
                            "delta_weight": delta_w,
                            "expected_direction": "INCREASE" if cardinality >= 2 else "INVARIANT_1.0",
                            "passed": passed,
                        })

        return {
            "property": "quality_monotonicity",
            "total_trials": len(trials),
            "passed_trials": sum(1 for t in trials if t["passed"]),
            "status": "PASS" if all_passed else "FAIL",
            "trials": trials,
        }

    def evaluate_reliability_monotonicity_fixture(self) -> Dict[str, Any]:
        """
        Evaluates d w_i / d R_i in an isolated mathematical fixture.
        Because R_i is a frozen global prior in production, this isolated fixture evaluates
        the mathematical logit kernel z_i = alpha*C + beta*R - gamma*U + eta*Q under beta=1.5 > 0.
        """
        trials: List[Dict[str, Any]] = []
        all_passed = True

        alpha = self.coefficients.alpha
        beta = self.coefficients.beta
        gamma = self.coefficients.gamma
        eta = self.coefficients.eta

        for regime in ACTIVE_REGIMES:
            active_mods = [m for m in MODALITIES if MODALITY_SYMBOLS[m] in regime]
            cardinality = len(active_mods)

            for target_mod in active_mods:
                for base_r in INTERIOR_GRID:
                    for step in PERTURBATION_STEPS:
                        new_r = min(1.0, base_r + step)
                        if new_r <= base_r:
                            continue

                        # Mathematical routing fixture with arbitrary R
                        def compute_weights_fixture(r_val: float) -> Dict[str, float]:
                            logits = {}
                            for m in active_mods:
                                c = 0.5
                                u = 0.2
                                q = 0.8
                                r = r_val if m == target_mod else 0.8
                                logits[m] = alpha * c + beta * r - gamma * u + eta * q
                            max_z = max(logits.values())
                            exps = {m: math.exp(logits[m] - max_z) for m in active_mods}
                            sum_e = sum(exps.values())
                            return {m: exps[m] / sum_e for m in active_mods}

                        w_before = compute_weights_fixture(base_r)[target_mod]
                        w_after = compute_weights_fixture(new_r)[target_mod]
                        delta_w = w_after - w_before

                        if cardinality >= 2:
                            passed = delta_w > STRICT_MONOTONIC_EPS
                        else:
                            passed = abs(w_after - 1.0) < FLOAT_TOLERANCE and abs(delta_w) < FLOAT_TOLERANCE

                        if not passed:
                            all_passed = False

                        trials.append({
                            "regime": regime,
                            "target_modality": target_mod,
                            "cardinality": cardinality,
                            "attribute": "reliability",
                            "val_before": base_r,
                            "val_after": new_r,
                            "weight_before": w_before,
                            "weight_after": w_after,
                            "delta_weight": delta_w,
                            "expected_direction": "INCREASE" if cardinality >= 2 else "INVARIANT_1.0",
                            "passed": passed,
                        })

        return {
            "property": "reliability_monotonicity_fixture",
            "total_trials": len(trials),
            "passed_trials": sum(1 for t in trials if t["passed"]),
            "status": "PASS" if all_passed else "FAIL",
            "trials": trials,
        }

    def run_all(self) -> Dict[str, Any]:
        """Executes all 4 isolated monotonicity evaluations."""
        conf_res = self.evaluate_confidence_monotonicity()
        unc_res = self.evaluate_uncertainty_monotonicity()
        qual_res = self.evaluate_quality_monotonicity()
        rel_res = self.evaluate_reliability_monotonicity_fixture()

        all_passed = all(
            r["status"] == "PASS" for r in [conf_res, unc_res, qual_res, rel_res]
        )

        return {
            "all_passed": all_passed,
            "confidence": conf_res,
            "uncertainty": unc_res,
            "quality": qual_res,
            "reliability": rel_res,
            "total_trials": conf_res["total_trials"] + unc_res["total_trials"] + qual_res["total_trials"] + rel_res["total_trials"],
        }
