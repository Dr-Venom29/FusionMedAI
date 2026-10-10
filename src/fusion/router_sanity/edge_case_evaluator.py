"""
FusionMedAI - Phase C11.15: Edge Case Evaluator
Verifies boundary conditions, error handling, and robust input sanitization:
1. All modalities unavailable (NO_MODALITY_AVAILABLE fail-closed)
2. Inactive channel value perturbation invariance (masked corruption resistance)
3. Comprehensive rejection of invalid channel inputs (NaN, Inf, out-of-range, reliability mismatch, quality-on-inactive)
4. Comprehensive rejection of invalid routing coefficients (NaN, Inf, out-of-bounds)
"""

from typing import Dict, List, Tuple, Any, Optional
import math
import numpy as np

from src.fusion.router.acarau_router import ACARAUv2Router
from src.fusion.router.router_input import (
    RouterInput,
    ModalityChannelInput,
    RouterContractValidationError,
    FROZEN_RELIABILITY_MAP,
)
from src.fusion.router.coefficients import (
    RouterCoefficients,
    RouterCoefficientError,
)
from src.fusion.router_sanity.sanity_config import FLOAT_TOLERANCE, ROUTER_COEFFS_REF


class EdgeCaseEvaluator:
    """Evaluates edge cases, fail-closed mechanics, and corruption invariance."""

    def __init__(self, coefficients: Optional[RouterCoefficients] = None):
        self.coefficients = coefficients or RouterCoefficients(
            alpha=ROUTER_COEFFS_REF["alpha"],
            beta=ROUTER_COEFFS_REF["beta"],
            gamma=ROUTER_COEFFS_REF["gamma"],
            eta=ROUTER_COEFFS_REF["eta"],
        )
        self.router = ACARAUv2Router(coefficients=self.coefficients)

    def check_empty_modalities_fail_closed(self) -> Dict[str, Any]:
        """
        Verifies that when all modalities are unavailable (A = empty),
        the router safely rejects and returns NO_MODALITY_AVAILABLE status,
        assigning 0.0 weights to all channels.
        """
        retina = ModalityChannelInput(
            modality="retina", confidence=0.0, uncertainty=1.0, quality=0.0, availability=False, reliability=FROZEN_RELIABILITY_MAP["retina"]
        )
        foot = ModalityChannelInput(
            modality="foot", confidence=0.0, uncertainty=1.0, quality=0.0, availability=False, reliability=FROZEN_RELIABILITY_MAP["foot"]
        )
        clinical = ModalityChannelInput(
            modality="clinical", confidence=0.0, uncertainty=1.0, quality=0.0, availability=False, reliability=FROZEN_RELIABILITY_MAP["clinical"]
        )

        empty_inp = RouterInput(retina=retina, foot=foot, clinical=clinical)
        res = self.router.route(empty_inp)

        status_ok = res.status == "NO_MODALITY_AVAILABLE"
        weights_zero = all(w == 0.0 for w in res.weights.values())
        num_active_zero = res.num_active == 0
        active_list_empty = len(res.active_modalities) == 0

        passed = status_ok and weights_zero and num_active_zero and active_list_empty

        return {
            "criterion_id": "S15-11",
            "passed": passed,
            "status": res.status,
            "weights": res.weights,
            "num_active": res.num_active,
            "active_modalities": res.active_modalities,
        }

    def check_masked_value_corruption_invariance(self) -> Dict[str, Any]:
        """
        Verifies that mutating inactive channel values (e.g. confidence/uncertainty)
        while availability=False has zero effect on active modality weights.
        """
        violations = []
        max_dev = 0.0

        # Test Regime FC (Retina is inactive, quality must be 0.0 when availability=False)
        base_retina = ModalityChannelInput(modality="retina", confidence=0.5, uncertainty=0.2, quality=0.0, availability=False, reliability=FROZEN_RELIABILITY_MAP["retina"])
        foot = ModalityChannelInput(modality="foot", confidence=0.7, uncertainty=0.15, quality=0.85, availability=True, reliability=FROZEN_RELIABILITY_MAP["foot"])
        clinical = ModalityChannelInput(modality="clinical", confidence=0.4, uncertainty=0.25, quality=0.9, availability=True, reliability=FROZEN_RELIABILITY_MAP["clinical"])

        base_inp = RouterInput(retina=base_retina, foot=foot, clinical=clinical)
        base_res = self.router.route(base_inp)

        # Mutate inactive retina values across valid range
        corruption_scenarios = [
            {"confidence": 0.0, "uncertainty": 1.0},
            {"confidence": 1.0, "uncertainty": 0.0},
            {"confidence": 0.99, "uncertainty": 0.01},
            {"confidence": 0.01, "uncertainty": 0.99},
        ]

        for sc in corruption_scenarios:
            corrupt_retina = ModalityChannelInput(
                modality="retina",
                confidence=sc["confidence"],
                uncertainty=sc["uncertainty"],
                quality=0.0,
                availability=False,
                reliability=FROZEN_RELIABILITY_MAP["retina"],
            )
            corrupt_inp = RouterInput(retina=corrupt_retina, foot=foot, clinical=clinical)
            corrupt_res = self.router.route(corrupt_inp)

            for m in ["foot", "clinical"]:
                dev = abs(corrupt_res.weights[m] - base_res.weights[m])
                if dev > max_dev:
                    max_dev = dev
                if dev > 1e-12:
                    violations.append({
                        "scenario": sc,
                        "modality": m,
                        "base_w": base_res.weights[m],
                        "corrupt_w": corrupt_res.weights[m],
                        "dev": dev,
                    })

            if corrupt_res.weights["retina"] != 0.0:
                violations.append({
                    "scenario": sc,
                    "modality": "retina",
                    "weight": corrupt_res.weights["retina"],
                    "issue": "inactive_weight_leak",
                })

        passed = len(violations) == 0 and max_dev < 1e-12
        return {
            "criterion_id": "S15-12",
            "passed": passed,
            "scenarios_tested": len(corruption_scenarios),
            "max_active_weight_dev": max_dev,
            "violation_count": len(violations),
            "violations": violations,
        }

    def check_input_validation_rejection(self) -> Dict[str, Any]:
        """
        Verifies that invalid input values across all dimensions (NaN, Inf, out-of-range,
        quality on inactive channels, reliability mismatch, and coefficient errors)
        are properly rejected by contracts with precise exception types.
        """
        rejection_cases = []

        def try_case(case_name: str, fn, expected_exceptions=(RouterContractValidationError, RouterCoefficientError)) -> None:
            try:
                fn()
                rejection_cases.append({"case": case_name, "rejected": False, "passed": False, "reason": "No exception raised"})
            except expected_exceptions as e:
                rejection_cases.append({"case": case_name, "rejected": True, "passed": True, "exception_type": type(e).__name__})
            except Exception as e:
                rejection_cases.append({"case": case_name, "rejected": False, "passed": False, "reason": f"Unexpected exception {type(e).__name__}: {str(e)}"})

        # 1. Confidence checks
        try_case("nan_confidence", lambda: ModalityChannelInput(
            modality="retina", confidence=float("nan"), uncertainty=0.1, quality=0.8, availability=True, reliability=FROZEN_RELIABILITY_MAP["retina"]
        ), expected_exceptions=(RouterContractValidationError,))
        try_case("inf_confidence", lambda: ModalityChannelInput(
            modality="retina", confidence=float("inf"), uncertainty=0.1, quality=0.8, availability=True, reliability=FROZEN_RELIABILITY_MAP["retina"]
        ), expected_exceptions=(RouterContractValidationError,))
        try_case("confidence_above_one", lambda: ModalityChannelInput(
            modality="retina", confidence=1.05, uncertainty=0.1, quality=0.8, availability=True, reliability=FROZEN_RELIABILITY_MAP["retina"]
        ), expected_exceptions=(RouterContractValidationError,))
        try_case("confidence_below_zero", lambda: ModalityChannelInput(
            modality="retina", confidence=-0.05, uncertainty=0.1, quality=0.8, availability=True, reliability=FROZEN_RELIABILITY_MAP["retina"]
        ), expected_exceptions=(RouterContractValidationError,))

        # 2. Uncertainty checks
        try_case("nan_uncertainty", lambda: ModalityChannelInput(
            modality="foot", confidence=0.5, uncertainty=float("nan"), quality=0.8, availability=True, reliability=FROZEN_RELIABILITY_MAP["foot"]
        ), expected_exceptions=(RouterContractValidationError,))
        try_case("inf_uncertainty", lambda: ModalityChannelInput(
            modality="foot", confidence=0.5, uncertainty=float("inf"), quality=0.8, availability=True, reliability=FROZEN_RELIABILITY_MAP["foot"]
        ), expected_exceptions=(RouterContractValidationError,))
        try_case("uncertainty_above_one", lambda: ModalityChannelInput(
            modality="foot", confidence=0.5, uncertainty=1.20, quality=0.8, availability=True, reliability=FROZEN_RELIABILITY_MAP["foot"]
        ), expected_exceptions=(RouterContractValidationError,))
        try_case("uncertainty_below_zero", lambda: ModalityChannelInput(
            modality="foot", confidence=0.5, uncertainty=-0.10, quality=0.8, availability=True, reliability=FROZEN_RELIABILITY_MAP["foot"]
        ), expected_exceptions=(RouterContractValidationError,))

        # 3. Quality checks
        try_case("nan_quality", lambda: ModalityChannelInput(
            modality="clinical", confidence=0.5, uncertainty=0.1, quality=float("nan"), availability=True, reliability=FROZEN_RELIABILITY_MAP["clinical"]
        ), expected_exceptions=(RouterContractValidationError,))
        try_case("inf_quality", lambda: ModalityChannelInput(
            modality="clinical", confidence=0.5, uncertainty=0.1, quality=float("inf"), availability=True, reliability=FROZEN_RELIABILITY_MAP["clinical"]
        ), expected_exceptions=(RouterContractValidationError,))
        try_case("quality_above_one", lambda: ModalityChannelInput(
            modality="clinical", confidence=0.5, uncertainty=0.1, quality=1.5, availability=True, reliability=FROZEN_RELIABILITY_MAP["clinical"]
        ), expected_exceptions=(RouterContractValidationError,))
        try_case("quality_below_zero", lambda: ModalityChannelInput(
            modality="clinical", confidence=0.5, uncertainty=0.1, quality=-0.2, availability=True, reliability=FROZEN_RELIABILITY_MAP["clinical"]
        ), expected_exceptions=(RouterContractValidationError,))
        try_case("quality_non_zero_when_inactive", lambda: ModalityChannelInput(
            modality="clinical", confidence=0.0, uncertainty=1.0, quality=0.8, availability=False, reliability=FROZEN_RELIABILITY_MAP["clinical"]
        ), expected_exceptions=(RouterContractValidationError,))

        # 4. Reliability checks
        try_case("reliability_mismatch", lambda: ModalityChannelInput(
            modality="retina", confidence=0.5, uncertainty=0.1, quality=0.8, availability=True, reliability=0.50
        ), expected_exceptions=(RouterContractValidationError,))
        try_case("nan_reliability", lambda: ModalityChannelInput(
            modality="retina", confidence=0.5, uncertainty=0.1, quality=0.8, availability=True, reliability=float("nan")
        ), expected_exceptions=(RouterContractValidationError,))

        # 5. Router Coefficient checks
        try_case("nan_alpha_coefficient", lambda: RouterCoefficients(alpha=float("nan"), beta=1.5, gamma=1.0, eta=0.5), expected_exceptions=(RouterCoefficientError,))
        try_case("inf_beta_coefficient", lambda: RouterCoefficients(alpha=1.0, beta=float("inf"), gamma=1.0, eta=0.5), expected_exceptions=(RouterCoefficientError,))
        try_case("negative_gamma_coefficient", lambda: RouterCoefficients(alpha=1.0, beta=1.5, gamma=-0.5, eta=0.5), expected_exceptions=(RouterCoefficientError,))
        try_case("out_of_bounds_eta_coefficient", lambda: RouterCoefficients(alpha=1.0, beta=1.5, gamma=1.0, eta=6.0), expected_exceptions=(RouterCoefficientError,))

        all_passed = all(c["passed"] for c in rejection_cases)
        return {
            "all_invalid_inputs_rejected": all_passed,
            "total_cases": len(rejection_cases),
            "passed_cases": sum(1 for c in rejection_cases if c["passed"]),
            "cases": rejection_cases,
        }

    def run_all_edge_case_checks(self) -> Dict[str, Any]:
        """Runs complete edge case and validation test suite."""
        empty_res = self.check_empty_modalities_fail_closed()
        masked_res = self.check_masked_value_corruption_invariance()
        valid_res = self.check_input_validation_rejection()

        all_passed = empty_res["passed"] and masked_res["passed"] and valid_res["all_invalid_inputs_rejected"]

        return {
            "all_edge_cases_passed": all_passed,
            "empty_modality_handling": empty_res,
            "masked_value_invariance": masked_res,
            "input_validation_rejection": valid_res,
        }
