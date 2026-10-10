"""
FusionMedAI - Phase C11.15: Router Invariant Checker
Verifies foundational mathematical invariants of the ACARA-U dynamic router:
1. Simplex conservation (sum w_i = 1.0, w_i >= 0) via production ACARAUv2Router.route()
2. Hard availability masking (A_i = 0 => w_i = 0.0 exact) via production ACARAUv2Router.route()
3. Softmax shift invariance:
   - End-to-end production route() invariance under uniform attribute logit shifts
   - Reference normalization kernel invariance under extreme scalar logit shifts [-500, +500]
4. Reference Normalization Kernel Permutation Invariance:
   - Order independence across all channel permutations in softmax normalization
5. Numerical stability under extreme valid boundary inputs and reference extreme-differential logits
6. Determinism (exact bitwise reproducibility across repeated production route calls)
"""

from typing import Dict, List, Tuple, Any, Optional
import math
import itertools
import numpy as np

from src.fusion.router.acarau_router import ACARAUv2Router
from src.fusion.router.router_input import (
    RouterInput,
    ModalityChannelInput,
    FROZEN_RELIABILITY_MAP,
)
from src.fusion.router.coefficients import RouterCoefficients
from src.fusion.router_sanity.sanity_config import (
    FLOAT_TOLERANCE,
    SHIFT_INVARIANCE_TOLERANCE,
    ROUTER_COEFFS_REF,
    ACTIVE_REGIMES,
)


class RouterInvariantChecker:
    """Evaluates and reports on mathematical invariants of the ACARA-U routing engine."""

    def __init__(self, coefficients: Optional[RouterCoefficients] = None):
        self.coefficients = coefficients or RouterCoefficients(
            alpha=ROUTER_COEFFS_REF["alpha"],
            beta=ROUTER_COEFFS_REF["beta"],
            gamma=ROUTER_COEFFS_REF["gamma"],
            eta=ROUTER_COEFFS_REF["eta"],
        )
        self.router = ACARAUv2Router(coefficients=self.coefficients)

    def _create_sample_input(
        self,
        c_vals: Tuple[float, float, float] = (0.7, 0.5, 0.3),
        u_vals: Tuple[float, float, float] = (0.1, 0.2, 0.05),
        q_vals: Tuple[float, float, float] = (0.9, 0.8, 0.95),
        avail: Tuple[bool, bool, bool] = (True, True, True),
    ) -> RouterInput:
        """Helper to create a canonical RouterInput with specified channel values conforming to contract."""
        retina = ModalityChannelInput(
            modality="retina",
            confidence=c_vals[0],
            uncertainty=u_vals[0],
            quality=q_vals[0] if avail[0] else 0.0,
            availability=avail[0],
            reliability=FROZEN_RELIABILITY_MAP["retina"],
        )
        foot = ModalityChannelInput(
            modality="foot",
            confidence=c_vals[1],
            uncertainty=u_vals[1],
            quality=q_vals[1] if avail[1] else 0.0,
            availability=avail[1],
            reliability=FROZEN_RELIABILITY_MAP["foot"],
        )
        clinical = ModalityChannelInput(
            modality="clinical",
            confidence=c_vals[2],
            uncertainty=u_vals[2],
            quality=q_vals[2] if avail[2] else 0.0,
            availability=avail[2],
            reliability=FROZEN_RELIABILITY_MAP["clinical"],
        )
        return RouterInput(
            retina=retina,
            foot=foot,
            clinical=clinical,
        )

    def check_simplex_conservation(self, n_trials: int = 100, seed: int = 42) -> Dict[str, Any]:
        """
        Verifies sum(w_i) == 1.0 within FLOAT_TOLERANCE and w_i >= 0.0 for all active regimes
        directly through production ACARAUv2Router.route().
        """
        rng = np.random.RandomState(seed)
        violations = []
        max_sum_dev = 0.0
        min_weight_observed = 1.0

        for regime in ACTIVE_REGIMES:
            r_avail = "R" in regime
            f_avail = "F" in regime
            c_avail = "C" in regime

            for trial in range(n_trials):
                c_vals = (float(rng.uniform(0.0, 1.0)), float(rng.uniform(0.0, 1.0)), float(rng.uniform(0.0, 1.0)))
                u_vals = (float(rng.uniform(0.0, 1.0)), float(rng.uniform(0.0, 1.0)), float(rng.uniform(0.0, 1.0)))
                q_vals = (float(rng.uniform(0.0, 1.0)), float(rng.uniform(0.0, 1.0)), float(rng.uniform(0.0, 1.0)))

                inp = self._create_sample_input(
                    c_vals=c_vals, u_vals=u_vals, q_vals=q_vals, avail=(r_avail, f_avail, c_avail)
                )
                res = self.router.route(inp)

                w_sum = sum(res.weights.values())
                dev = abs(w_sum - 1.0)
                if dev > max_sum_dev:
                    max_sum_dev = dev

                for m, w in res.weights.items():
                    if w < min_weight_observed:
                        min_weight_observed = w
                    if w < -FLOAT_TOLERANCE:
                        violations.append({
                            "regime": regime,
                            "trial": trial,
                            "modality": m,
                            "weight": w,
                            "issue": "negative_weight",
                        })

                if dev > 1e-10:
                    violations.append({
                        "regime": regime,
                        "trial": trial,
                        "sum": w_sum,
                        "deviation": dev,
                        "issue": "simplex_sum_violation",
                    })

        passed = len(violations) == 0 and max_sum_dev < 1e-10 and min_weight_observed >= 0.0
        return {
            "criterion_id": "S15-06",
            "passed": passed,
            "n_trials_per_regime": n_trials,
            "total_evaluations": len(ACTIVE_REGIMES) * n_trials,
            "max_sum_deviation": max_sum_dev,
            "min_weight_observed": min_weight_observed,
            "violation_count": len(violations),
            "violations": violations[:5],
        }

    def check_hard_masking(self, n_trials: int = 50, seed: int = 42) -> Dict[str, Any]:
        """
        Verifies that inactive modalities (A_i = 0) strictly receive w_i == 0.0 exact
        directly through production ACARAUv2Router.route().
        """
        rng = np.random.RandomState(seed)
        violations = []
        partial_regimes = [r for r in ACTIVE_REGIMES if len(r) < 3]

        for regime in partial_regimes:
            r_avail = "R" in regime
            f_avail = "F" in regime
            c_avail = "C" in regime

            for trial in range(n_trials):
                c_vals = (float(rng.uniform(0.0, 1.0)), float(rng.uniform(0.0, 1.0)), float(rng.uniform(0.0, 1.0)))
                u_vals = (float(rng.uniform(0.0, 1.0)), float(rng.uniform(0.0, 1.0)), float(rng.uniform(0.0, 1.0)))
                q_vals = (float(rng.uniform(0.0, 1.0)), float(rng.uniform(0.0, 1.0)), float(rng.uniform(0.0, 1.0)))

                inp = self._create_sample_input(
                    c_vals=c_vals, u_vals=u_vals, q_vals=q_vals, avail=(r_avail, f_avail, c_avail)
                )
                res = self.router.route(inp)

                if not r_avail and res.weights["retina"] != 0.0:
                    violations.append({"regime": regime, "modality": "retina", "weight": res.weights["retina"]})
                if not f_avail and res.weights["foot"] != 0.0:
                    violations.append({"regime": regime, "modality": "foot", "weight": res.weights["foot"]})
                if not c_avail and res.weights["clinical"] != 0.0:
                    violations.append({"regime": regime, "modality": "clinical", "weight": res.weights["clinical"]})

        passed = len(violations) == 0
        return {
            "criterion_id": "S15-07",
            "passed": passed,
            "partial_regimes_tested": partial_regimes,
            "total_evaluations": len(partial_regimes) * n_trials,
            "violation_count": len(violations),
            "violations": violations[:5],
        }

    def check_shift_invariance(self) -> Dict[str, Any]:
        """
        Verifies mathematical softmax shift invariance:
        1. End-to-end production route() invariance under uniform attribute shifts.
        2. Reference normalization kernel invariance under extreme scalar offsets [-500, +500].
        """
        violations = []
        max_shift_dev = 0.0

        # Part A: End-to-end production route() shift check
        # Uniform shift in confidence: Delta C = +0.10 across all active channels
        base_inp = self._create_sample_input(
            c_vals=(0.4, 0.5, 0.6),
            u_vals=(0.1, 0.2, 0.15),
            q_vals=(0.8, 0.9, 0.85),
            avail=(True, True, True),
        )
        base_res = self.router.route(base_inp)

        # Uniform additive shift in confidence (+0.10, +0.10, +0.10)
        shifted_inp = self._create_sample_input(
            c_vals=(0.5, 0.6, 0.7),
            u_vals=(0.1, 0.2, 0.15),
            q_vals=(0.8, 0.9, 0.85),
            avail=(True, True, True),
        )
        shifted_res = self.router.route(shifted_inp)

        route_violations = []
        max_route_dev = 0.0
        for m in base_inp.available_modalities:
            w_base = base_res.weights[m]
            w_shift = shifted_res.weights[m]
            dev = abs(w_shift - w_base)
            if dev > max_route_dev:
                max_route_dev = dev
            if dev > max_shift_dev:
                max_shift_dev = dev
            if dev > 1e-12:
                route_violations.append({"test": "route_uniform_shift", "modality": m, "dev": dev})
                violations.append({"test": "route_uniform_shift", "modality": m, "dev": dev})

        production_route_shift_passed = bool(len(route_violations) == 0 and max_route_dev < 1e-12)

        # Part B: Reference logit normalization kernel evaluation under large offsets [-500, 500]
        shifts = [-500.0, -100.0, -50.0, -10.0, -1.0, 0.0, 1.0, 10.0, 50.0, 100.0, 500.0]
        raw_logits, _ = self.router.compute_logits(base_inp)

        kernel_violations = []
        max_kernel_dev = 0.0
        for c in shifts:
            shifted_raw = {m: raw_logits[m] + c for m in base_inp.available_modalities}
            max_score = max(shifted_raw.values())
            exp_scores = {m: math.exp(shifted_raw[m] - max_score) for m in base_inp.available_modalities}
            sum_exp = sum(exp_scores.values())
            shifted_weights = {m: exp_scores[m] / sum_exp for m in base_inp.available_modalities}

            for m in base_inp.available_modalities:
                w_base = base_res.weights[m]
                w_shift = shifted_weights[m]
                dev = abs(w_shift - w_base)
                if dev > max_kernel_dev:
                    max_kernel_dev = dev
                if dev > max_shift_dev:
                    max_shift_dev = dev
                if dev > 1e-12:
                    kernel_violations.append({"test": "kernel_extreme_shift", "shift": c, "modality": m, "dev": dev})
                    violations.append({"test": "kernel_extreme_shift", "shift": c, "modality": m, "dev": dev})

        reference_kernel_shift_passed = bool(len(kernel_violations) == 0 and max_kernel_dev < 1e-12)
        passed = bool(production_route_shift_passed and reference_kernel_shift_passed)

        return {
            "criterion_id": "S15-08",
            "passed": passed,
            "production_route_shift_passed": production_route_shift_passed,
            "max_route_shift_dev": max_route_dev,
            "reference_kernel_shift_passed": reference_kernel_shift_passed,
            "shifts_tested": shifts,
            "max_shift_deviation": max_shift_dev,
            "violation_count": len(violations),
            "violations": violations[:5],
            "scope": "Combined production route() uniform shift and reference normalization kernel shifts",
        }

    def check_kernel_permutation_invariance(self) -> Dict[str, Any]:
        """
        Verifies reference normalization kernel order independence across all channel permutations.
        Permuting raw logit vectors across all 3! = 6 permutations in the softmax normalization
        function preserves exact output weight values.
        """
        violations = []
        max_perm_dev = 0.0

        base_inp = self._create_sample_input(
            c_vals=(0.8, 0.4, 0.6),
            u_vals=(0.1, 0.3, 0.2),
            q_vals=(0.95, 0.7, 0.85),
            avail=(True, True, True),
        )
        base_res = self.router.route(base_inp)
        raw_logits, _ = self.router.compute_logits(base_inp)

        orderings = list(itertools.permutations(["retina", "foot", "clinical"]))
        kernel_violations = []
        for perm in orderings:
            perm_scores = [raw_logits[m] for m in perm]
            max_score = max(perm_scores)
            exp_scores = {m: math.exp(raw_logits[m] - max_score) for m in perm}
            sum_exp = sum(exp_scores.values())
            perm_weights = {m: exp_scores[m] / sum_exp for m in perm}

            for m in ["retina", "foot", "clinical"]:
                w_ref = base_res.weights[m]
                w_perm = perm_weights[m]
                dev = abs(w_ref - w_perm)
                if dev > max_perm_dev:
                    max_perm_dev = dev
                if dev > 1e-12:
                    kernel_violations.append({"permutation": perm, "modality": m, "dev": dev})
                    violations.append({"test": "kernel_perm", "permutation": perm, "modality": m, "dev": dev})

        reference_kernel_perm_passed = bool(len(kernel_violations) == 0 and max_perm_dev < 1e-12)
        passed = reference_kernel_perm_passed

        return {
            "criterion_id": "S15-09",
            "passed": passed,
            "reference_kernel_perm_passed": reference_kernel_perm_passed,
            "permutations_tested": len(orderings),
            "max_permutation_deviation": max_perm_dev,
            "violation_count": len(violations),
            "scope": "Reference normalization kernel order independence across all channel permutations",
        }

    def check_numerical_stability(self) -> Dict[str, Any]:
        """
        Verifies numerical stability of the router:
        1. Production ACARAUv2Router.route() evaluated across 5 extreme valid boundary configurations.
        2. Reference softmax normalization kernel evaluated under extreme logit differentials [+500, 0, -500].
        """
        test_cases = [
            {"name": "extreme_positive", "c": (1.0, 1.0, 1.0), "u": (0.0, 0.0, 0.0), "q": (1.0, 1.0, 1.0)},
            {"name": "extreme_negative", "c": (0.0, 0.0, 0.0), "u": (1.0, 1.0, 1.0), "q": (0.0, 0.0, 0.0)},
            {"name": "high_disparity", "c": (1.0, 0.0, 0.5), "u": (0.0, 1.0, 0.5), "q": (1.0, 0.0, 0.5)},
            {"name": "all_zeros", "c": (0.0, 0.0, 0.0), "u": (0.0, 0.0, 0.0), "q": (0.0, 0.0, 0.0)},
            {"name": "all_ones", "c": (1.0, 1.0, 1.0), "u": (1.0, 1.0, 1.0), "q": (1.0, 1.0, 1.0)},
        ]

        results = []
        route_boundary_passed = True

        for case in test_cases:
            inp = self._create_sample_input(
                c_vals=case["c"],
                u_vals=case["u"],
                q_vals=case["q"],
                avail=(True, True, True),
            )
            res = self.router.route(inp)

            all_finite = all(math.isfinite(w) for w in res.weights.values())
            sum_valid = abs(sum(res.weights.values()) - 1.0) < 1e-10
            in_range = all(0.0 <= w <= 1.0 for w in res.weights.values())
            case_passed = bool(all_finite and sum_valid and in_range)

            if not case_passed:
                route_boundary_passed = False

            results.append({
                "case_name": case["name"],
                "passed": case_passed,
                "weights": res.weights,
                "entropy": res.routing_entropy,
            })

        # Reference extreme logit differential (+500.0, 0.0, -500.0) testing subtraction-by-max stability
        extreme_logits = [500.0, 0.0, -500.0]
        max_z = max(extreme_logits)
        exp_z = [math.exp(z - max_z) for z in extreme_logits]
        sum_e = sum(exp_z)
        extreme_weights = [e / sum_e for e in exp_z]
        extreme_finite = all(math.isfinite(w) for w in extreme_weights)
        extreme_sum = abs(sum(extreme_weights) - 1.0) < 1e-10
        extreme_dominant = extreme_weights[0] > 0.999999

        reference_kernel_extreme_logits_passed = bool(extreme_finite and extreme_sum and extreme_dominant)
        passed = bool(route_boundary_passed and reference_kernel_extreme_logits_passed)

        return {
            "criterion_id": "S15-10",
            "passed": passed,
            "production_route_boundary_passed": route_boundary_passed,
            "test_cases": results,
            "reference_kernel_extreme_logits_passed": reference_kernel_extreme_logits_passed,
            "reference_kernel_extreme_weights": extreme_weights,
            "scope": "Production route() boundary inputs and reference normalization kernel extreme differentials",
        }

    def check_determinism(self, n_repeats: int = 20) -> Dict[str, Any]:
        """
        Verifies that repeating identical inputs yields bitwise identical weights
        directly through production ACARAUv2Router.route().
        """
        inp = self._create_sample_input(
            c_vals=(0.77, 0.44, 0.88),
            u_vals=(0.12, 0.34, 0.06),
            q_vals=(0.91, 0.82, 0.99),
            avail=(True, True, True),
        )
        first_res = self.router.route(inp)
        first_weights = first_res.weights

        all_identical = True
        for _ in range(n_repeats - 1):
            res = self.router.route(inp)
            if res.weights != first_weights:
                all_identical = False
                break

        return {
            "passed": all_identical,
            "n_repeats": n_repeats,
            "weights": first_weights,
        }

    def run_all_invariant_checks(self) -> Dict[str, Any]:
        """Runs the complete suite of mathematical invariant checks and aggregates results."""
        simplex = self.check_simplex_conservation()
        masking = self.check_hard_masking()
        shift = self.check_shift_invariance()
        perm = self.check_kernel_permutation_invariance()
        stability = self.check_numerical_stability()
        determinism = self.check_determinism()

        all_passed = all([
            simplex["passed"],
            masking["passed"],
            shift["passed"],
            perm["passed"],
            stability["passed"],
            determinism["passed"],
        ])

        return {
            "all_invariants_passed": all_passed,
            "simplex_conservation": simplex,
            "hard_availability_masking": masking,
            "softmax_shift_invariance": shift,
            "modality_permutation_invariance": perm,
            "numerical_stability": stability,
            "determinism": determinism,
        }
