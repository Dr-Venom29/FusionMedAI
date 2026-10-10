"""
FusionMedAI - Phase C11.14: Deep Verification Suite (12 Verification Gates)
Performs independent re-execution, mathematical invariant verification,
exhaustive field-by-field numerical reconciliation, and cryptographic certification.
"""

import sys
import json
import math
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

# Robust repo root discovery
def get_repo_root() -> Path:
    curr = Path(__file__).resolve()
    for parent in [curr] + list(curr.parents):
        if (parent / "requirements.txt").is_file() and (parent / "src").is_dir():
            return parent
    if len(curr.parents) >= 4:
        return curr.parents[3]
    return Path.cwd()

REPO_ROOT = get_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fusion.dcri.cohort_loader import load_frozen_cohort
from src.fusion.dcri_policy.policy_config import (
    ALPHA_REF,
    BETA_REF,
    GAMMA_REF,
    ETA_REF,
    DELTA_FROZEN,
    TAU_1_DEFAULT,
    TAU_2_DEFAULT,
    TAU_1_GRID,
    TAU_2_GRID,
    ACTIVE_REGIMES,
    ALL_REGIMES,
    N_PACKETS,
    SEED,
    N_BOOTSTRAPS,
    FLOAT_TOLERANCE,
    RESIDUAL_TOLERANCE,
    ACTION_ROUTINE_REVIEW,
    ACTION_ADDITIONAL_ASSESSMENT,
    ACTION_ESCALATION,
    ACTION_NAMES,
    ACTION_KEYS,
)
from src.fusion.dcri_policy.policy_engine import PolicyEvaluator, assign_action
from src.fusion.dcri_policy.threshold_sweeper import ThresholdSensitivitySweeper
from src.fusion.dcri_policy.regime_policy_evaluator import RegimePolicyEvaluator
from src.fusion.dcri_policy.robustness_evaluator import RobustnessEvaluator
from src.fusion.dcri_policy.policy_stats import PolicyStatisticalAnalyzer


def compute_sha256_file(filepath: Path) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


class PolicyVerificationSuite:
    """
    Executes 12 Deep Verification Gates for Phase C11.14.
    """

    EXPECTED_PACKET_MANIFEST_SHA256 = "1ff37fbeeba33feae7bf14d57d0df4a10df48ac789edeedf9dbd2fa8121382bb"
    EXPECTED_PACKET_STREAM_DIGEST = "491da3d908652915e51f744d03c89c55db776549a8b18d6c32771c12d18e96bd"

    def __init__(self, repo_root: Path = REPO_ROOT):
        self.repo_root = repo_root
        self.results_dir = repo_root / "experiments" / "fusion" / "dcri_policy_analysis" / "results"
        self.cohort = load_frozen_cohort(repo_root=self.repo_root, n_packets=N_PACKETS, seed=SEED)
        self.evaluator = PolicyEvaluator(delta=DELTA_FROZEN)

    def assertEqual(self, a, b, msg: str = ""):
        if a != b:
            raise AssertionError(f"Assertion failed: {a} != {b}. {msg}")

    def assertAlmostEqual(self, a: float, b: float, tol: float = 1e-12, msg: str = ""):
        if abs(a - b) > tol:
            raise AssertionError(f"Assertion failed: |{a} - {b}| = {abs(a - b)} > {tol}. {msg}")

    def assert_raises_value_error(self, func, *args, **kwargs):
        try:
            func(*args, **kwargs)
        except ValueError:
            return
        except Exception as e:
            raise AssertionError(f"Expected ValueError from {func.__name__}, got {type(e).__name__}: {e}")
        else:
            raise AssertionError(f"Expected ValueError from {func.__name__}, but function returned normally.")

    def run_all_gates(self) -> bool:
        print("=" * 80)
        print("PHASE C11.14: DCRI DECISION POLICY ANALYSIS — 12 DEEP VERIFICATION GATES")
        print("=" * 80)
        print()

        gates = [
            ("Gate 01 - Frozen Cohort Provenance & Cryptographic Packet Stream Digest", self.verify_gate_01),
            ("Gate 02 - Frozen Reference Router Theta_0 Configuration", self.verify_gate_02),
            ("Gate 03 - Frozen Delta Multiplier & Policy Threshold Specification", self.verify_gate_03),
            ("Gate 04 - Exact Analytical DCRI Equation Recomputation (< 1e-14)", self.verify_gate_04),
            ("Gate 05 - Decision Action Tier Assignment & Boundary Equality Invariants", self.verify_gate_05),
            ("Gate 06 - Negative DCRI Safety & Tier 0 Assignment Invariant", self.verify_gate_06),
            ("Gate 07 - Monotonic Non-Inflationary Reclassification (Zero Upgrades)", self.verify_gate_07),
            ("Gate 08 - All 7 Active Modality Regimes Evaluated with Simplex Conservation", self.verify_gate_08),
            ("Gate 09 - Real Router EMPTY Regime Fail-Closed Contract", self.verify_gate_09),
            ("Gate 10 - Robustness Perturbation Stability & Perturbation Family Audit", self.verify_gate_10),
            ("Gate 11 - Full Paired Bootstrap & Wilson Statistical Audit (B=1000)", self.verify_gate_11),
            ("Gate 12 - Exhaustive 5-Artifact Field Numerical & Cryptographic Manifest Certification", self.verify_gate_12),
        ]

        all_passed = True
        for gate_name, gate_fn in gates:
            try:
                gate_fn()
                print(f"{gate_name}: PASS")
            except Exception as e:
                print(f"{gate_name}: FAIL -> {e}")
                all_passed = False

        print()
        print("=" * 80)
        if all_passed:
            print("PHASE C11.14 VERIFICATION SUMMARY: 12/12 GATES PASSED")
            print(">>> [PASS] ALL 12 DEEP VERIFICATION GATES PASSED. C11.14 IS FULLY SEALED.")
        else:
            print("PHASE C11.14 VERIFICATION SUMMARY: GATE FAILURES DETECTED")
        print("=" * 80)
        return all_passed

    def verify_gate_01(self):
        """Gate 01: Verify cohort provenance, size N=500, sequential IDs, and cryptographic digests."""
        if len(self.cohort) != 500:
            raise ValueError(f"Expected N=500 cohort, got {len(self.cohort)}")
        ids = [p.packet_id for p in self.cohort]
        if len(set(ids)) != 500:
            raise ValueError("Duplicate packet IDs detected in cohort")
        expected_first = "PACKET_0000"
        expected_last = "PACKET_0499"
        if ids[0] != expected_first or ids[-1] != expected_last:
            raise ValueError(f"Cohort IDs not sequential: first={ids[0]}, last={ids[-1]}")

        # Cryptographic verification of underlying frozen packet manifest
        manifest_path = self.repo_root / "experiments" / "fusion" / "dcri" / "packet_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Cohort manifest missing at {manifest_path}")
        computed_manifest_hash = compute_sha256_file(manifest_path)
        if computed_manifest_hash != self.EXPECTED_PACKET_MANIFEST_SHA256:
            raise ValueError(
                f"Cohort manifest SHA-256 mismatch:\n"
                f"  Computed: {computed_manifest_hash}\n"
                f"  Expected: {self.EXPECTED_PACKET_MANIFEST_SHA256}"
            )

        # Content stream digest verification
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        h = hashlib.sha256()
        for pkt in manifest_data.get("sample_packets", []):
            pkt_str = json.dumps(pkt, sort_keys=True)
            h.update(pkt_str.encode("utf-8"))
        computed_stream_digest = h.hexdigest()
        if computed_stream_digest != self.EXPECTED_PACKET_STREAM_DIGEST:
            raise ValueError(
                f"Cohort packet stream digest mismatch:\n"
                f"  Computed: {computed_stream_digest}\n"
                f"  Expected: {self.EXPECTED_PACKET_STREAM_DIGEST}"
            )

    def verify_gate_02(self):
        """Gate 02: Verify router reference coefficients Theta_0."""
        coeff = self.evaluator.coeff
        if (coeff.alpha, coeff.beta, coeff.gamma, coeff.eta) != (1.0, 1.5, 1.0, 0.5):
            raise ValueError(f"Router coefficients corrupted: {coeff}")

    def verify_gate_03(self):
        """Gate 03: Verify frozen delta=0.10 and nominal thresholds (0.20, 0.40), plus grids."""
        if self.evaluator.delta != 0.10:
            raise ValueError(f"Expected delta=0.10, got {self.evaluator.delta}")
        if (TAU_1_DEFAULT, TAU_2_DEFAULT) != (0.20, 0.40):
            raise ValueError(f"Default thresholds corrupted: {(TAU_1_DEFAULT, TAU_2_DEFAULT)}")
        if len(TAU_1_GRID) != 5 or len(TAU_2_GRID) != 5:
            raise ValueError(f"Threshold grid dimensions corrupted: {len(TAU_1_GRID)}x{len(TAU_2_GRID)}")

    def verify_gate_04(self):
        """Gate 04: Exact analytical DCRI formula recomputation (< 1e-14)."""
        states = self.evaluator.precompute_packet_base_state(self.cohort)
        for s in states:
            expected_dcri = s["r_fusion"] - 0.10 * s["u_sum"]
            diff = abs(s["dcri"] - expected_dcri)
            if diff > 1e-14:
                raise ValueError(f"Packet {s['packet_id']} DCRI recomputation error: {diff}")

    def verify_gate_05(self):
        """Gate 05: Verify boundary equality, action mapping rules, and rejection of non-finite inputs."""
        self.assertEqual(assign_action(0.199999, 0.20, 0.40), ACTION_ROUTINE_REVIEW)
        self.assertEqual(assign_action(0.200000, 0.20, 0.40), ACTION_ADDITIONAL_ASSESSMENT)
        self.assertEqual(assign_action(0.399999, 0.20, 0.40), ACTION_ADDITIONAL_ASSESSMENT)
        self.assertEqual(assign_action(0.400000, 0.20, 0.40), ACTION_ESCALATION)

        # Reject non-finite scores (Defect B fixed with dedicated assertion helper)
        for val in [float("nan"), float("inf"), float("-inf")]:
            self.assert_raises_value_error(assign_action, val, 0.20, 0.40)

        # Reject non-finite thresholds
        for val in [float("nan"), float("inf"), float("-inf")]:
            self.assert_raises_value_error(assign_action, 0.25, val, 0.40)
            self.assert_raises_value_error(assign_action, 0.25, 0.20, val)

        # Reject inverted or equal thresholds
        self.assert_raises_value_error(assign_action, 0.25, 0.40, 0.20)
        self.assert_raises_value_error(assign_action, 0.25, 0.30, 0.30)

    def verify_gate_06(self):
        """Gate 06: Negative DCRI safety and tier 0 mapping."""
        states = self.evaluator.precompute_packet_base_state(self.cohort)
        res = self.evaluator.evaluate_policies(states, TAU_1_DEFAULT, TAU_2_DEFAULT)
        neg_count = sum(1 for p in res.per_packet_details if p["dcri"] < 0.0)
        if neg_count != 37:
            raise ValueError(f"Expected 37 negative DCRI packets at delta=0.10, found {neg_count}")
        for p in res.per_packet_details:
            if p["dcri"] < 0.0:
                if p["action_dcri"] != ACTION_ROUTINE_REVIEW:
                    raise ValueError(f"Negative DCRI packet {p['packet_id']} not assigned to Routine Review")

    def verify_gate_07(self):
        """Gate 07: Monotonic non-inflationary reclassification (zero upgrades)."""
        states = self.evaluator.precompute_packet_base_state(self.cohort)
        res = self.evaluator.evaluate_policies(states, TAU_1_DEFAULT, TAU_2_DEFAULT)
        if res.upgraded_count != 0:
            raise ValueError(f"Detected {res.upgraded_count} upward reclassifications (violates non-inflation invariant)")
        if res.reclassification_count != res.downgraded_count:
            raise ValueError(f"Reclassification mismatch: total={res.reclassification_count}, downgrades={res.downgraded_count}")
        if res.reclassification_count != 92:
            raise ValueError(f"Expected 92 reclassifications, found {res.reclassification_count}")

    def verify_gate_08(self):
        """Gate 08: All 7 active regimes evaluated with simplex conservation."""
        reg_eval = RegimePolicyEvaluator(delta=0.10, tau_1=0.20, tau_2=0.40)
        reg_res = reg_eval.evaluate_regimes(self.cohort)
        for reg in ACTIVE_REGIMES:
            if reg not in reg_res:
                raise ValueError(f"Missing regime {reg}")
            r = reg_res[reg]
            if r["n_packets"] != 500:
                raise ValueError(f"Regime {reg} packet count != 500")
            # Tier percentage sums
            pct_a = sum(r["policy_a_dcri"]["percentages"].values())
            pct_b = sum(r["policy_b_fused_risk"]["percentages"].values())
            if abs(pct_a - 100.0) > 1e-2 or abs(pct_b - 100.0) > 1e-2:
                raise ValueError(f"Regime {reg} percentages do not sum to 100: A={pct_a}, B={pct_b}")
            # Non-inflationary reclassifications
            if r["reclassifications"]["total_count"] != r["reclassifications"]["downgraded_count"]:
                raise ValueError(f"Regime {reg} total reclassifications != downgrades")

    def verify_gate_09(self):
        """Gate 09: Real router EMPTY regime fail-closed contract."""
        reg_eval = RegimePolicyEvaluator(delta=0.10, tau_1=0.20, tau_2=0.40)
        reg_res = reg_eval.evaluate_regimes(self.cohort)
        empty = reg_res["EMPTY"]
        if empty["fail_closed_status"] != "NO_MODALITY_AVAILABLE":
            raise ValueError(f"EMPTY regime status is not fail-closed: {empty['fail_closed_status']}")
        if empty["decision_output_available"]:
            raise ValueError("EMPTY regime incorrectly reported decision_output_available = True")
        if empty["sentinel_risk"] != 0.0:
            raise ValueError(f"EMPTY regime sentinel risk != 0.0: {empty['sentinel_risk']}")
        if not all(w == 0.0 for w in empty["routing_weights"].values()):
            raise ValueError(f"EMPTY regime non-zero weights: {empty['routing_weights']}")

    def verify_gate_10(self):
        """Gate 10: Robustness perturbation stability across both uncertainty scaling and threshold jitter."""
        rob_eval = RobustnessEvaluator(delta=0.10, tau_1=0.20, tau_2=0.40)
        states = self.evaluator.precompute_packet_base_state(self.cohort)

        # 1. Uncertainty scaling perturbation family
        unc_res = rob_eval.evaluate_uncertainty_scaling(states)
        if len(unc_res) != 5:
            raise ValueError(f"Expected 5 uncertainty scaling evaluations, got {len(unc_res)}")
        neg_counts = [r["negative_count"] for r in unc_res]
        if not all(neg_counts[i] <= neg_counts[i+1] for i in range(len(neg_counts)-1)):
            raise ValueError(f"Negative counts not monotonic under uncertainty scaling: {neg_counts}")
        reclass_counts = [r["reclassification_count"] for r in unc_res]
        if not all(reclass_counts[i] <= reclass_counts[i+1] for i in range(len(reclass_counts)-1)):
            raise ValueError(f"Reclassification counts not monotonic under uncertainty scaling: {reclass_counts}")

        # 2. Threshold boundary jitter perturbation family
        jit_res = rob_eval.evaluate_threshold_jitter(states)
        if len(jit_res) != 5:
            raise ValueError(f"Expected 5 threshold jitter evaluations, got {len(jit_res)}")
        for r in jit_res:
            pct_a_sum = sum(r["policy_a_percentages"].values())
            pct_b_sum = sum(r["policy_b_percentages"].values())
            if abs(pct_a_sum - 100.0) > 1e-2 or abs(pct_b_sum - 100.0) > 1e-2:
                raise ValueError(f"Jitter tier percentages do not sum to 100: A={pct_a_sum}, B={pct_b_sum}")

    def verify_gate_11(self):
        """Gate 11: Full Paired Bootstrap & Wilson Statistical Audit (B=1000)."""
        states = self.evaluator.precompute_packet_base_state(self.cohort)
        res = self.evaluator.evaluate_policies(states, TAU_1_DEFAULT, TAU_2_DEFAULT)
        stats_eval = PolicyStatisticalAnalyzer(n_bootstraps=1000, seed=115)
        stat_res = stats_eval.compute_paired_statistics(states, res.per_packet_details)

        # Verify score discount CI
        diff_ci = stat_res["score_discount"]["bootstrap_95_ci"]
        if diff_ci[0] <= 0.0:
            raise ValueError(f"Bootstrap score difference CI crosses zero: {diff_ci}")

        # Verify reclassification Wilson CI
        reclass_wilson = stat_res["reclassifications"]["wilson_95_ci"]
        if abs(reclass_wilson[0] - 0.152489) > 1e-4 or abs(reclass_wilson[1] - 0.220329) > 1e-4:
            raise ValueError(f"Wilson CI mismatch: {reclass_wilson}")

        # Verify escalation workload reduction
        esc_red = stat_res["escalation_reductions"]
        if esc_red["observed_rate"] != 0.072:
            raise ValueError(f"Expected 0.072 observed escalation reduction rate, got {esc_red['observed_rate']}")

    def verify_gate_12(self):
        """Gate 12: Exhaustive 5-artifact field-by-field numerical reconciliation, schema check, and manifest hash certification."""
        manifest_path = self.results_dir / "freeze_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest missing at {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        expected_artifacts = [
            "policy_config.json",
            "threshold_results.json",
            "regime_results.json",
            "robustness_results.json",
            "statistical_results.json",
        ]

        # 1. Cryptographic Checksum, Exact File Size & Manifest Metadata Validation
        self.assertEqual(set(manifest.keys()), {"phase", "manifest_version", "cohort_seed", "n_packets", "delta_frozen", "artifacts"}, "Manifest top-level schema keys mismatch")
        self.assertEqual(manifest.get("phase"), "C11.14", "Manifest phase mismatch")
        self.assertEqual(manifest.get("manifest_version"), "1.0.0", "Manifest version mismatch")
        self.assertEqual(manifest.get("cohort_seed"), SEED, "Manifest cohort_seed mismatch")
        self.assertEqual(manifest.get("n_packets"), N_PACKETS, "Manifest n_packets mismatch")
        self.assertEqual(manifest.get("delta_frozen"), DELTA_FROZEN, "Manifest delta_frozen mismatch")
        self.assertEqual(set(manifest["artifacts"].keys()), set(expected_artifacts), "Manifest artifact keys mismatch")

        for filename in expected_artifacts:
            filepath = self.results_dir / filename
            if not filepath.is_file():
                raise FileNotFoundError(f"Artifact file missing: {filepath}")
            
            entry = manifest["artifacts"][filename]
            self.assertEqual(set(entry.keys()), {"sha256", "size_bytes"}, f"Manifest artifact entry schema mismatch for {filename}")

            # Verify exact recorded file size
            actual_size = filepath.stat().st_size
            recorded_size = entry["size_bytes"]
            if actual_size != recorded_size:
                raise ValueError(
                    f"File size mismatch for {filename}:\n"
                    f"  Actual on disk: {actual_size} bytes\n"
                    f"  Recorded in manifest: {recorded_size} bytes"
                )

            # Verify cryptographic SHA-256 hash
            computed_hash = compute_sha256_file(filepath)
            recorded_hash = entry["sha256"]
            if computed_hash != recorded_hash:
                raise ValueError(
                    f"SHA-256 hash mismatch for {filename}:\n"
                    f"  Computed: {computed_hash}\n"
                    f"  Recorded: {recorded_hash}"
                )

        # 2. Independent In-Memory Recomputations
        states = self.evaluator.precompute_packet_base_state(self.cohort)
        sweeper = ThresholdSensitivitySweeper(delta=DELTA_FROZEN)
        recomputed_thresh = sweeper.sweep(states)

        reg_eval = RegimePolicyEvaluator(delta=DELTA_FROZEN, tau_1=TAU_1_DEFAULT, tau_2=TAU_2_DEFAULT)
        recomputed_reg = reg_eval.evaluate_regimes(self.cohort)

        rob_eval = RobustnessEvaluator(delta=DELTA_FROZEN, tau_1=TAU_1_DEFAULT, tau_2=TAU_2_DEFAULT)
        recomputed_unc_scaling = rob_eval.evaluate_uncertainty_scaling(states)
        recomputed_thresh_jitter = rob_eval.evaluate_threshold_jitter(states)

        nom_res = self.evaluator.evaluate_policies(states, TAU_1_DEFAULT, TAU_2_DEFAULT)
        stats_eval = PolicyStatisticalAnalyzer(n_bootstraps=1000, seed=115)
        recomputed_stat = stats_eval.compute_paired_statistics(states, nom_res.per_packet_details)

        # 3. Exhaustive Field-by-Field Reconciliation against on-disk JSON
        # A. policy_config.json
        with open(self.results_dir / "policy_config.json", "r", encoding="utf-8") as f:
            disk_config = json.load(f)
        self.assertEqual(
            set(disk_config.keys()),
            {"phase", "experiment_title", "frozen_parameters", "policy_definitions", "regimes_evaluated", "status"},
            "policy_config schema keys mismatch",
        )
        self.assertEqual(disk_config["phase"], "C11.14", "policy_config phase mismatch")
        self.assertEqual(
            disk_config["experiment_title"],
            "Decision-Policy Sensitivity and Operating-Behavior Analysis of Frozen DCRI Index",
            "policy_config experiment_title mismatch",
        )
        self.assertEqual(disk_config["status"], "FROZEN", "policy_config status mismatch")
        frozen_params = disk_config["frozen_parameters"]
        self.assertEqual(
            set(frozen_params.keys()),
            {"delta", "delta_provisional_historical", "router_coefficients", "cohort_size", "seed", "n_bootstraps"},
            "frozen_parameters schema mismatch",
        )
        self.assertEqual(frozen_params["delta"], DELTA_FROZEN, "delta mismatch")
        self.assertEqual(frozen_params["delta_provisional_historical"], 0.2, "historical delta mismatch")
        self.assertEqual(
            set(frozen_params["router_coefficients"].keys()),
            {"alpha", "beta", "gamma", "eta"},
            "router_coefficients schema mismatch",
        )
        self.assertEqual(
            frozen_params["router_coefficients"],
            {"alpha": ALPHA_REF, "beta": BETA_REF, "gamma": GAMMA_REF, "eta": ETA_REF},
            "router coefficients mismatch",
        )
        self.assertEqual(frozen_params["cohort_size"], N_PACKETS, "cohort_size mismatch")
        self.assertEqual(frozen_params["seed"], SEED, "seed mismatch")
        self.assertEqual(frozen_params["n_bootstraps"], N_BOOTSTRAPS, "n_bootstraps mismatch")
        policy_defs = disk_config["policy_definitions"]
        self.assertEqual(
            set(policy_defs.keys()),
            {"action_categories", "action_keys", "standard_thresholds", "threshold_sweep_grids"},
            "policy_definitions schema mismatch",
        )
        self.assertEqual(
            policy_defs["action_categories"],
            {str(k): v for k, v in ACTION_NAMES.items()},
            "action_categories mismatch",
        )
        self.assertEqual(
            policy_defs["action_keys"],
            {str(k): v for k, v in ACTION_KEYS.items()},
            "action_keys mismatch",
        )
        self.assertEqual(
            set(policy_defs["standard_thresholds"].keys()),
            {"tau_1", "tau_2"},
            "standard_thresholds schema mismatch",
        )
        self.assertEqual(
            policy_defs["standard_thresholds"],
            {"tau_1": TAU_1_DEFAULT, "tau_2": TAU_2_DEFAULT},
            "standard_thresholds mismatch",
        )
        self.assertEqual(
            set(policy_defs["threshold_sweep_grids"].keys()),
            {"tau_1_grid", "tau_2_grid"},
            "threshold_sweep_grids schema mismatch",
        )
        self.assertEqual(
            policy_defs["threshold_sweep_grids"]["tau_1_grid"],
            list(TAU_1_GRID),
            "tau_1_grid mismatch",
        )
        self.assertEqual(
            policy_defs["threshold_sweep_grids"]["tau_2_grid"],
            list(TAU_2_GRID),
            "tau_2_grid mismatch",
        )
        self.assertEqual(disk_config["regimes_evaluated"], list(ALL_REGIMES), "regimes_evaluated mismatch")

        # B. threshold_results.json
        with open(self.results_dir / "threshold_results.json", "r", encoding="utf-8") as f:
            disk_thresh = json.load(f)
        self.assertEqual(
            set(disk_thresh.keys()),
            {"phase", "primary_operating_point", "threshold_sweep_results", "summary_insights"},
            "threshold_results schema keys mismatch",
        )
        self.assertEqual(disk_thresh["phase"], "C11.14", "threshold_results phase mismatch")

        # Reconcile primary operating point exhaustively
        disk_nom = disk_thresh["primary_operating_point"]
        self.assertEqual(
            set(disk_nom.keys()),
            {"tau_1", "tau_2", "delta", "n_packets", "policy_a_dcri", "policy_b_fused_risk", "transition_matrix_fused_to_dcri", "reclassifications"},
            "primary_operating_point schema mismatch",
        )
        self.assertEqual(disk_nom["tau_1"], TAU_1_DEFAULT, "primary op tau_1 mismatch")
        self.assertEqual(disk_nom["tau_2"], TAU_2_DEFAULT, "primary op tau_2 mismatch")
        self.assertEqual(disk_nom["delta"], DELTA_FROZEN, "primary op delta mismatch")
        self.assertEqual(disk_nom["n_packets"], N_PACKETS, "primary op n_packets mismatch")
        self.assertEqual(set(disk_nom["policy_a_dcri"].keys()), {"counts", "percentages"}, "policy_a_dcri schema mismatch")
        self.assertEqual(set(disk_nom["policy_a_dcri"]["counts"].keys()), {"routine_review", "additional_assessment", "escalation"}, "policy_a counts keys mismatch")
        self.assertEqual(set(disk_nom["policy_a_dcri"]["percentages"].keys()), {"routine_review", "additional_assessment", "escalation"}, "policy_a percentages keys mismatch")
        self.assertEqual(disk_nom["policy_a_dcri"]["counts"], nom_res.policy_a_counts, "primary op policy_a counts mismatch")
        self.assertEqual(disk_nom["policy_a_dcri"]["percentages"], nom_res.policy_a_pcts, "primary op policy_a percentages mismatch")
        
        self.assertEqual(set(disk_nom["policy_b_fused_risk"].keys()), {"counts", "percentages"}, "policy_b_fused_risk schema mismatch")
        self.assertEqual(set(disk_nom["policy_b_fused_risk"]["counts"].keys()), {"routine_review", "additional_assessment", "escalation"}, "policy_b counts keys mismatch")
        self.assertEqual(set(disk_nom["policy_b_fused_risk"]["percentages"].keys()), {"routine_review", "additional_assessment", "escalation"}, "policy_b percentages keys mismatch")
        self.assertEqual(disk_nom["policy_b_fused_risk"]["counts"], nom_res.policy_b_counts, "primary op policy_b counts mismatch")
        self.assertEqual(disk_nom["policy_b_fused_risk"]["percentages"], nom_res.policy_b_pcts, "primary op policy_b percentages mismatch")
        self.assertEqual(disk_nom["transition_matrix_fused_to_dcri"], nom_res.transition_matrix, "primary op transition matrix mismatch")
        
        disk_reclass = disk_nom["reclassifications"]
        self.assertEqual(
            set(disk_reclass.keys()),
            {"total_reclassified_count", "total_reclassified_rate", "downgraded_count", "downgraded_rate", "upgraded_count", "upgraded_rate", "escalation_reduction_count", "escalation_reduction_rate"},
            "reclassifications schema mismatch",
        )
        self.assertEqual(disk_reclass["total_reclassified_count"], nom_res.reclassification_count, "reclassified count mismatch")
        self.assertAlmostEqual(disk_reclass["total_reclassified_rate"], nom_res.reclassification_rate, 1e-6, "reclassified rate mismatch")
        self.assertEqual(disk_reclass["downgraded_count"], nom_res.downgraded_count, "downgraded count mismatch")
        self.assertAlmostEqual(disk_reclass["downgraded_rate"], nom_res.downgraded_rate, 1e-6, "downgraded rate mismatch")
        self.assertEqual(disk_reclass["upgraded_count"], nom_res.upgraded_count, "upgraded count mismatch")
        self.assertAlmostEqual(disk_reclass["upgraded_rate"], nom_res.upgraded_rate, 1e-6, "upgraded rate mismatch")
        self.assertEqual(disk_reclass["escalation_reduction_count"], nom_res.escalation_reduction_count, "escalation reduction count mismatch")
        self.assertAlmostEqual(disk_reclass["escalation_reduction_rate"], nom_res.escalation_reduction_rate, 1e-6, "escalation reduction rate mismatch")

        # Reconcile all 25 threshold sweep results exhaustively
        self.assertEqual(len(disk_thresh["threshold_sweep_results"]), 25, "threshold sweep count != 25")
        self.assertEqual(len(recomputed_thresh), 25, "recomputed threshold sweep count != 25")
        for d_entry, r_entry in zip(disk_thresh["threshold_sweep_results"], recomputed_thresh):
            self.assertEqual(
                set(d_entry.keys()),
                {"tau_1", "tau_2", "delta", "policy_a_dcri", "policy_b_fused_risk", "transition_matrix", "reclassifications"},
                "threshold sweep entry schema mismatch",
            )
            self.assertEqual((d_entry["tau_1"], d_entry["tau_2"]), (r_entry["tau_1"], r_entry["tau_2"]), "Threshold grid pair mismatch")
            self.assertEqual(d_entry["delta"], r_entry["delta"], "Threshold sweep delta mismatch")
            self.assertEqual(set(d_entry["policy_a_dcri"].keys()), {"counts", "percentages"}, "sweep policy_a schema mismatch")
            self.assertEqual(d_entry["policy_a_dcri"]["counts"], r_entry["policy_a_dcri"]["counts"], "Threshold sweep policy_a counts mismatch")
            self.assertEqual(d_entry["policy_a_dcri"]["percentages"], r_entry["policy_a_dcri"]["percentages"], "Threshold sweep policy_a percentages mismatch")
            self.assertEqual(set(d_entry["policy_b_fused_risk"].keys()), {"counts", "percentages"}, "sweep policy_b schema mismatch")
            self.assertEqual(d_entry["policy_b_fused_risk"]["counts"], r_entry["policy_b_fused_risk"]["counts"], "Threshold sweep policy_b counts mismatch")
            self.assertEqual(d_entry["policy_b_fused_risk"]["percentages"], r_entry["policy_b_fused_risk"]["percentages"], "Threshold sweep policy_b percentages mismatch")
            self.assertEqual(d_entry["transition_matrix"], r_entry["transition_matrix"], "Threshold sweep transition matrix mismatch")
            self.assertEqual(
                set(d_entry["reclassifications"].keys()),
                {"total_count", "rate", "downgraded_count", "downgraded_rate", "upgraded_count", "upgraded_rate", "escalation_reduction_count", "escalation_reduction_rate"},
                "sweep reclassifications schema mismatch",
            )
            self.assertEqual(d_entry["reclassifications"], r_entry["reclassifications"], "Threshold sweep reclassifications dict mismatch")

        # Reconcile summary_insights exhaustively
        disk_insights = disk_thresh["summary_insights"]
        self.assertEqual(
            set(disk_insights.keys()),
            {"nominal_reclassification_rate", "nominal_downgraded_rate", "nominal_upgraded_rate", "nominal_escalation_reduction_rate", "monotonic_downgrade_invariant"},
            "summary_insights schema keys mismatch",
        )
        self.assertAlmostEqual(disk_insights["nominal_reclassification_rate"], nom_res.reclassification_rate, 1e-6, "summary_insights reclass rate mismatch")
        self.assertAlmostEqual(disk_insights["nominal_downgraded_rate"], nom_res.downgraded_rate, 1e-6, "summary_insights downgraded rate mismatch")
        self.assertAlmostEqual(disk_insights["nominal_upgraded_rate"], nom_res.upgraded_rate, 1e-6, "summary_insights upgraded rate mismatch")
        self.assertAlmostEqual(disk_insights["nominal_escalation_reduction_rate"], nom_res.escalation_reduction_rate, 1e-6, "summary_insights escalation red rate mismatch")
        self.assertEqual(disk_insights["monotonic_downgrade_invariant"], (nom_res.upgraded_count == 0), "summary_insights monotonic invariant mismatch")

        # C. regime_results.json
        with open(self.results_dir / "regime_results.json", "r", encoding="utf-8") as f:
            disk_reg = json.load(f)
        self.assertEqual(
            set(disk_reg.keys()),
            {"phase", "delta", "standard_thresholds", "regime_results"},
            "regime_results schema keys mismatch",
        )
        self.assertEqual(disk_reg["phase"], "C11.14", "regime_results phase mismatch")
        self.assertEqual(disk_reg["delta"], DELTA_FROZEN, "regime delta mismatch")
        self.assertEqual(
            set(disk_reg["standard_thresholds"].keys()),
            {"tau_1", "tau_2"},
            "regime standard_thresholds schema mismatch",
        )
        self.assertEqual(
            disk_reg["standard_thresholds"],
            {"tau_1": TAU_1_DEFAULT, "tau_2": TAU_2_DEFAULT},
            "regime standard thresholds mismatch",
        )
        self.assertEqual(
            set(disk_reg["regime_results"].keys()),
            set(ALL_REGIMES),
            "Exact regime keys mismatch",
        )
        for reg_key in ACTIVE_REGIMES:
            d_r = disk_reg["regime_results"][reg_key]
            r_r = recomputed_reg[reg_key]
            self.assertEqual(
                set(d_r.keys()),
                {"regime", "n_packets", "cardinality", "mean_r_fusion", "mean_u_sum", "mean_dcri", "negative_count", "negative_rate", "policy_a_dcri", "policy_b_fused_risk", "reclassifications", "transition_matrix"},
                f"Regime {reg_key} entry schema mismatch",
            )
            self.assertEqual(d_r["regime"], reg_key, f"Regime {reg_key} name mismatch")
            self.assertEqual(d_r["n_packets"], N_PACKETS, f"Regime {reg_key} n_packets mismatch")
            self.assertEqual(d_r["cardinality"], len(reg_key), f"Regime {reg_key} cardinality mismatch")
            self.assertAlmostEqual(d_r["mean_r_fusion"], r_r["mean_r_fusion"], 1e-6, f"Regime {reg_key} mean_r_fusion mismatch")
            self.assertAlmostEqual(d_r["mean_u_sum"], r_r["mean_u_sum"], 1e-6, f"Regime {reg_key} mean_u_sum mismatch")
            self.assertAlmostEqual(d_r["mean_dcri"], r_r["mean_dcri"], 1e-6, f"Regime {reg_key} mean_dcri mismatch")
            self.assertEqual(d_r["negative_count"], r_r["negative_count"], f"Regime {reg_key} negative_count mismatch")
            self.assertAlmostEqual(d_r["negative_rate"], r_r["negative_rate"], 1e-6, f"Regime {reg_key} negative_rate mismatch")
            self.assertEqual(set(d_r["policy_a_dcri"].keys()), {"counts", "percentages"}, f"Regime {reg_key} policy_a schema mismatch")
            self.assertEqual(d_r["policy_a_dcri"], r_r["policy_a_dcri"], f"Regime {reg_key} policy_a_dcri mismatch")
            self.assertEqual(set(d_r["policy_b_fused_risk"].keys()), {"counts", "percentages"}, f"Regime {reg_key} policy_b schema mismatch")
            self.assertEqual(d_r["policy_b_fused_risk"], r_r["policy_b_fused_risk"], f"Regime {reg_key} policy_b_fused_risk mismatch")
            self.assertEqual(
                set(d_r["reclassifications"].keys()),
                {"total_count", "rate", "downgraded_count", "downgraded_rate", "escalation_reduction_count", "escalation_reduction_rate"},
                f"Regime {reg_key} reclassifications schema mismatch",
            )
            self.assertEqual(d_r["reclassifications"], r_r["reclassifications"], f"Regime {reg_key} reclassifications mismatch")
            self.assertEqual(d_r["transition_matrix"], r_r["transition_matrix"], f"Regime {reg_key} transition matrix mismatch")

        # Reconcile EMPTY regime
        d_empty = disk_reg["regime_results"]["EMPTY"]
        r_empty = recomputed_reg["EMPTY"]
        self.assertEqual(
            set(d_empty.keys()),
            {"regime", "n_packets", "cardinality", "fail_closed_status", "decision_output_available", "sentinel_risk", "routing_weights"},
            "EMPTY regime schema mismatch",
        )
        self.assertEqual(d_empty["regime"], "EMPTY", "EMPTY regime name mismatch")
        self.assertEqual(d_empty["n_packets"], N_PACKETS, "EMPTY regime n_packets mismatch")
        self.assertEqual(d_empty["cardinality"], 0, "EMPTY regime cardinality mismatch")
        self.assertEqual(d_empty["fail_closed_status"], r_empty["fail_closed_status"], "EMPTY fail_closed_status mismatch")
        self.assertEqual(d_empty["decision_output_available"], False, "EMPTY decision_output_available mismatch")
        self.assertEqual(d_empty["sentinel_risk"], 0.0, "EMPTY sentinel_risk mismatch")
        self.assertEqual(d_empty["routing_weights"], r_empty["routing_weights"], "EMPTY routing_weights mismatch")

        # D. robustness_results.json
        with open(self.results_dir / "robustness_results.json", "r", encoding="utf-8") as f:
            disk_rob = json.load(f)
        self.assertEqual(
            set(disk_rob.keys()),
            {"phase", "uncertainty_scaling_perturbation", "threshold_jitter_perturbation"},
            "robustness_results schema keys mismatch",
        )
        self.assertEqual(disk_rob["phase"], "C11.14", "robustness_results phase mismatch")
        
        # Explicit length validation before comparison
        self.assertEqual(len(disk_rob["uncertainty_scaling_perturbation"]), 5, "Uncertainty scaling list length != 5")
        self.assertEqual(len(recomputed_unc_scaling), 5, "Recomputed uncertainty scaling list length != 5")
        self.assertEqual(len(disk_rob["threshold_jitter_perturbation"]), 5, "Threshold jitter list length != 5")
        self.assertEqual(len(recomputed_thresh_jitter), 5, "Recomputed threshold jitter list length != 5")

        for d_u, r_u in zip(disk_rob["uncertainty_scaling_perturbation"], recomputed_unc_scaling):
            self.assertEqual(
                set(d_u.keys()),
                {"scale_factor", "n_packets", "negative_count", "negative_rate", "policy_a_counts", "policy_a_percentages", "reclassification_count", "reclassification_rate", "downgraded_count", "downgraded_rate"},
                "uncertainty scaling entry schema mismatch",
            )
            self.assertEqual(d_u["scale_factor"], r_u["scale_factor"], "Scale factor mismatch")
            self.assertEqual(d_u["n_packets"], N_PACKETS, "Uncertainty scaling n_packets mismatch")
            self.assertEqual(d_u["negative_count"], r_u["negative_count"], "Uncertainty scaling negative_count mismatch")
            self.assertAlmostEqual(d_u["negative_rate"], r_u["negative_rate"], 1e-6, "Uncertainty scaling negative_rate mismatch")
            self.assertEqual(set(d_u["policy_a_counts"].keys()), {"routine_review", "additional_assessment", "escalation"}, "uncertainty scaling counts schema mismatch")
            self.assertEqual(set(d_u["policy_a_percentages"].keys()), {"routine_review", "additional_assessment", "escalation"}, "uncertainty scaling pcts schema mismatch")
            self.assertEqual(d_u["policy_a_counts"], r_u["policy_a_counts"], "Uncertainty scaling policy_a_counts mismatch")
            self.assertEqual(d_u["policy_a_percentages"], r_u["policy_a_percentages"], "Uncertainty scaling policy_a_percentages mismatch")
            self.assertEqual(d_u["reclassification_count"], r_u["reclassification_count"], "Uncertainty scaling reclass count mismatch")
            self.assertAlmostEqual(d_u["reclassification_rate"], r_u["reclassification_rate"], 1e-6, "Uncertainty scaling reclass rate mismatch")
            self.assertEqual(d_u["downgraded_count"], r_u["downgraded_count"], "Uncertainty scaling downgraded count mismatch")
            self.assertAlmostEqual(d_u["downgraded_rate"], r_u["downgraded_rate"], 1e-6, "Uncertainty scaling downgraded rate mismatch")

        for d_j, r_j in zip(disk_rob["threshold_jitter_perturbation"], recomputed_thresh_jitter):
            self.assertEqual(
                set(d_j.keys()),
                {"jitter_offset", "tau_1", "tau_2", "policy_a_counts", "policy_a_percentages", "policy_b_counts", "policy_b_percentages", "reclassification_count", "reclassification_rate", "escalation_reduction_count", "escalation_reduction_rate"},
                "threshold jitter entry schema mismatch",
            )
            self.assertEqual(d_j["jitter_offset"], r_j["jitter_offset"], "Jitter offset mismatch")
            self.assertEqual(d_j["tau_1"], r_j["tau_1"], "Jitter tau_1 mismatch")
            self.assertEqual(d_j["tau_2"], r_j["tau_2"], "Jitter tau_2 mismatch")
            self.assertEqual(set(d_j["policy_a_counts"].keys()), {"routine_review", "additional_assessment", "escalation"}, "jitter policy_a_counts schema mismatch")
            self.assertEqual(set(d_j["policy_a_percentages"].keys()), {"routine_review", "additional_assessment", "escalation"}, "jitter policy_a_percentages schema mismatch")
            self.assertEqual(d_j["policy_a_counts"], r_j["policy_a_counts"], "Jitter policy_a counts mismatch")
            self.assertEqual(d_j["policy_a_percentages"], r_j["policy_a_percentages"], "Jitter policy_a percentages mismatch")
            self.assertEqual(set(d_j["policy_b_counts"].keys()), {"routine_review", "additional_assessment", "escalation"}, "jitter policy_b_counts schema mismatch")
            self.assertEqual(set(d_j["policy_b_percentages"].keys()), {"routine_review", "additional_assessment", "escalation"}, "jitter policy_b_percentages schema mismatch")
            self.assertEqual(d_j["policy_b_counts"], r_j["policy_b_counts"], "Jitter policy_b counts mismatch")
            self.assertEqual(d_j["policy_b_percentages"], r_j["policy_b_percentages"], "Jitter policy_b percentages mismatch")
            self.assertEqual(d_j["reclassification_count"], r_j["reclassification_count"], "Jitter reclass count mismatch")
            self.assertAlmostEqual(d_j["reclassification_rate"], r_j["reclassification_rate"], 1e-6, "Jitter reclass rate mismatch")
            self.assertEqual(d_j["escalation_reduction_count"], r_j["escalation_reduction_count"], "Jitter escalation reduction count mismatch")
            self.assertAlmostEqual(d_j["escalation_reduction_rate"], r_j["escalation_reduction_rate"], 1e-6, "Jitter escalation reduction rate mismatch")

        # E. statistical_results.json
        with open(self.results_dir / "statistical_results.json", "r", encoding="utf-8") as f:
            disk_stat = json.load(f)
        self.assertEqual(
            set(disk_stat.keys()),
            {"phase", "statistical_summary"},
            "statistical_results schema keys mismatch",
        )
        self.assertEqual(disk_stat["phase"], "C11.14", "statistical_results phase mismatch")
        d_summary = disk_stat["statistical_summary"]
        self.assertEqual(
            set(d_summary.keys()),
            {"n_packets", "seed", "n_bootstraps", "score_discount", "reclassifications", "downgrades", "escalation_reductions"},
            "statistical_summary schema keys mismatch",
        )
        self.assertEqual(d_summary["n_packets"], N_PACKETS, "stat n_packets mismatch")
        self.assertEqual(d_summary["seed"], SEED, "stat seed mismatch")
        self.assertEqual(d_summary["n_bootstraps"], N_BOOTSTRAPS, "stat n_bootstraps mismatch")

        # score_discount
        self.assertEqual(set(d_summary["score_discount"].keys()), {"observed_mean_difference", "bootstrap_mean", "bootstrap_std_error", "bootstrap_95_ci"}, "score_discount schema mismatch")
        self.assertAlmostEqual(d_summary["score_discount"]["observed_mean_difference"], recomputed_stat["score_discount"]["observed_mean_difference"], 1e-6, "score_discount obs diff mismatch")
        self.assertAlmostEqual(d_summary["score_discount"]["bootstrap_mean"], recomputed_stat["score_discount"]["bootstrap_mean"], 1e-6, "score_discount boot mean mismatch")
        self.assertAlmostEqual(d_summary["score_discount"]["bootstrap_std_error"], recomputed_stat["score_discount"]["bootstrap_std_error"], 1e-6, "score_discount boot se mismatch")
        self.assertEqual(d_summary["score_discount"]["bootstrap_95_ci"], recomputed_stat["score_discount"]["bootstrap_95_ci"], "score_discount CI mismatch")

        # reclassifications
        self.assertEqual(set(d_summary["reclassifications"].keys()), {"observed_rate", "bootstrap_mean_rate", "bootstrap_95_ci", "wilson_95_ci"}, "reclassifications schema mismatch")
        self.assertAlmostEqual(d_summary["reclassifications"]["observed_rate"], recomputed_stat["reclassifications"]["observed_rate"], 1e-6, "reclass obs rate mismatch")
        self.assertAlmostEqual(d_summary["reclassifications"]["bootstrap_mean_rate"], recomputed_stat["reclassifications"]["bootstrap_mean_rate"], 1e-6, "reclass boot mean mismatch")
        self.assertEqual(d_summary["reclassifications"]["bootstrap_95_ci"], recomputed_stat["reclassifications"]["bootstrap_95_ci"], "reclass boot CI mismatch")
        self.assertEqual(d_summary["reclassifications"]["wilson_95_ci"], recomputed_stat["reclassifications"]["wilson_95_ci"], "reclass wilson CI mismatch")

        # downgrades
        self.assertEqual(set(d_summary["downgrades"].keys()), {"observed_rate", "bootstrap_mean_rate", "bootstrap_95_ci", "wilson_95_ci"}, "downgrades schema mismatch")
        self.assertAlmostEqual(d_summary["downgrades"]["observed_rate"], recomputed_stat["downgrades"]["observed_rate"], 1e-6, "downgrades obs rate mismatch")
        self.assertAlmostEqual(d_summary["downgrades"]["bootstrap_mean_rate"], recomputed_stat["downgrades"]["bootstrap_mean_rate"], 1e-6, "downgrades boot mean mismatch")
        self.assertEqual(d_summary["downgrades"]["bootstrap_95_ci"], recomputed_stat["downgrades"]["bootstrap_95_ci"], "downgrades boot CI mismatch")
        self.assertEqual(d_summary["downgrades"]["wilson_95_ci"], recomputed_stat["downgrades"]["wilson_95_ci"], "downgrades wilson CI mismatch")

        # escalation_reductions
        self.assertEqual(set(d_summary["escalation_reductions"].keys()), {"observed_rate", "bootstrap_mean_rate", "bootstrap_95_ci", "wilson_95_ci"}, "escalation_reductions schema mismatch")
        self.assertAlmostEqual(d_summary["escalation_reductions"]["observed_rate"], recomputed_stat["escalation_reductions"]["observed_rate"], 1e-6, "esc red obs rate mismatch")
        self.assertAlmostEqual(d_summary["escalation_reductions"]["bootstrap_mean_rate"], recomputed_stat["escalation_reductions"]["bootstrap_mean_rate"], 1e-6, "esc red boot mean mismatch")
        self.assertEqual(d_summary["escalation_reductions"]["bootstrap_95_ci"], recomputed_stat["escalation_reductions"]["bootstrap_95_ci"], "esc red boot CI mismatch")
        self.assertEqual(d_summary["escalation_reductions"]["wilson_95_ci"], recomputed_stat["escalation_reductions"]["wilson_95_ci"], "esc red wilson CI mismatch")


if __name__ == "__main__":
    suite = PolicyVerificationSuite()
    success = suite.run_all_gates()
    sys.exit(0 if success else 1)
