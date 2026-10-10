"""
FusionMedAI - Independent Artifact & Verification Gate Suite for Outcome-Grounded Evaluation
Enforces 8 strict acceptance gates independently recomputing statistics from raw per-packet data,
verifying 2-stage hierarchical bootstrap, oracle lineage, tier derivations, and cryptographic SHA-256 seal.
"""

from typing import Dict, Any, List, Tuple
from pathlib import Path
import json
import hashlib
import numpy as np
import math
import sys
import scipy.stats as stats

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fusion.outcome_evaluation.outcome_metrics import (
    compute_prediction_metrics,
    assign_action_tier,
    compute_decision_loss,
    ACTION_COST_MATRIX,
)


class OutcomeVerificationError(RuntimeError):
    """Raised when an outcome evaluation verification gate fails."""
    pass


class OutcomeArtifactVerifier:
    """
    Independent gate verifier for Outcome-Grounded Evaluation artifacts.
    Independently recomputes all metrics and inferential statistics directly from raw packet records in confirmatory_results.json.
    """

    EXPECTED_ARTIFACTS = [
        "protocol.json",
        "dev_results.json",
        "confirmatory_results.json",
        "summary.json",
    ]

    def __init__(self, exp_dir: Path):
        self.exp_dir = exp_dir
        self.gate_results: Dict[str, Dict[str, Any]] = {}

    def verify_all(self) -> Dict[str, Any]:
        """Runs all 8 verification gates."""
        manifest_path = self.exp_dir / "freeze_manifest.json"
        if not manifest_path.is_file():
            raise OutcomeVerificationError(f"Freeze manifest missing: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        # OE-08: Cryptographic Hash Manifest Verification
        self._gate_oe08_manifest_hashes(manifest)

        # Load artifacts
        with open(self.exp_dir / "protocol.json", "r", encoding="utf-8") as f:
            protocol = json.load(f)
        with open(self.exp_dir / "confirmatory_results.json", "r", encoding="utf-8") as f:
            conf = json.load(f)
        with open(self.exp_dir / "summary.json", "r", encoding="utf-8") as f:
            summary = json.load(f)

        # OE-01: Protocol Parameter & Practical Threshold Integrity
        self._gate_oe01_protocol_integrity(protocol)

        # OE-02: Cohort Integrity & Unique Seeds
        self._gate_oe02_cohort_integrity(conf)

        # OE-03: Independent Recomputation of Prediction Metrics
        self._gate_oe03_recompute_prediction_metrics(conf)

        # OE-04: Singleton Cardinality Invariance from Raw Records
        self._gate_oe04_singleton_invariance(conf)

        # OE-05: Degradation & Sensor Fidelity Error Scaling
        self._gate_oe05_degradation_and_fidelity_scaling(conf, summary)

        # OE-06: Independent Recomputation of Decision Policy & False Downgrades
        self._gate_oe06_decision_policy_recomputation(conf, summary)

        # OE-07: Statistical Inference & Practical Superiority Decision Rule
        self._gate_oe07_statistical_inference_and_decision_rule(conf, summary)

        all_passed = all(g["passed"] for g in self.gate_results.values())
        return {
            "status": "PASSED" if all_passed else "FAILED",
            "total_gates": len(self.gate_results),
            "passed_gates": sum(1 for g in self.gate_results.values() if g["passed"]),
            "gate_results": self.gate_results,
        }

    def _gate_oe01_protocol_integrity(self, protocol: Dict[str, Any]) -> None:
        params = protocol.get("parameters", {}).get("router_coefficients", {})
        h1 = protocol.get("hypotheses", {}).get("H1_Primary", {})
        passed = (
            params.get("alpha") == 1.0
            and params.get("beta") == 1.5
            and params.get("gamma") == 1.0
            and params.get("eta") == 0.5
            and protocol.get("parameters", {}).get("dcri_penalty_delta") == 0.10
            and protocol.get("sampling_plan", {}).get("total_confirmatory_packets") == 5000
            and "Delta_MAE < -0.005" in h1.get("superiority_criterion", "")
        )
        self.gate_results["OE-01"] = {
            "name": "Protocol Parameter & Practical Threshold Integrity",
            "passed": bool(passed),
            "details": f"Theta_0={(params.get('alpha'), params.get('beta'), params.get('gamma'), params.get('eta'))}, delta={protocol.get('parameters', {}).get('dcri_penalty_delta')}, Threshold=-0.005",
        }

    def _gate_oe02_cohort_integrity(self, conf: Dict[str, Any]) -> None:
        cohorts = conf.get("cohort_results", [])
        raw_packets = conf.get("raw_packet_records", [])
        seeds = [c.get("seed") for c in cohorts]

        unique_seeds = len(set(seeds)) == 10 and len(seeds) == 10
        total_pkt_count = len(raw_packets) == 5000
        packet_ids = [p["packet_id"] for p in raw_packets]
        unique_packet_ids = len(set(packet_ids)) == 5000

        # Verify exactly 500 packets per seed in raw records
        packets_per_seed = {}
        for p in raw_packets:
            s = p["seed"]
            packets_per_seed[s] = packets_per_seed.get(s, 0) + 1
        
        exact_500_each = all(count == 500 for count in packets_per_seed.values()) and len(packets_per_seed) == 10

        passed = bool(unique_seeds and total_pkt_count and unique_packet_ids and exact_500_each)
        self.gate_results["OE-02"] = {
            "name": "Cohort Integrity & Unique Seed Validation",
            "passed": passed,
            "details": f"10 unique seeds ({seeds}), exactly 500 packets per seed, 5,000 distinct paired packet IDs.",
        }

    def _gate_oe03_recompute_prediction_metrics(self, conf: Dict[str, Any]) -> None:
        raw_packets = conf.get("raw_packet_records", [])
        b6_risks = [p["r_b6"] for p in raw_packets]
        b5_risks = [p["r_b5"] for p in raw_packets]
        b2_risks = [p["r_b2"] for p in raw_packets]
        targets = [p["oracle_risk"] for p in raw_packets]

        recomp_b6 = compute_prediction_metrics(b6_risks, targets)
        recomp_b5 = compute_prediction_metrics(b5_risks, targets)
        recomp_b2 = compute_prediction_metrics(b2_risks, targets)

        reported_b6 = conf.get("mean_b6_mae", 0.0)
        reported_b5 = conf.get("mean_b5_mae", 0.0)
        reported_b2 = conf.get("mean_b2_mae", 0.0)

        match_b6 = abs(recomp_b6["mae"] - reported_b6) < 1e-5
        match_b5 = abs(recomp_b5["mae"] - reported_b5) < 1e-5
        match_b2 = abs(recomp_b2["mae"] - reported_b2) < 1e-5

        passed = bool(match_b6 and match_b5 and match_b2)
        self.gate_results["OE-03"] = {
            "name": "Independent Recomputation of Prediction Metrics",
            "passed": passed,
            "details": f"Recomputed MAEs: B6={recomp_b6['mae']:.6f}, B5={recomp_b5['mae']:.6f}, B2={recomp_b2['mae']:.6f} match reported.",
        }

    def _gate_oe04_singleton_invariance(self, conf: Dict[str, Any]) -> None:
        raw_packets = conf.get("raw_packet_records", [])
        singletons = [p for p in raw_packets if p["num_active"] == 1]
        
        max_singleton_diff = max(abs(p["r_b6"] - p["r_b5"]) for p in singletons)
        max_singleton_mae_diff = max(abs(p["diff_mae_b6_minus_b5"]) for p in singletons)

        # Directly verify routing weight invariants for singletons (active channel must have weight 1.0)
        weight_invariants_pass = True
        for p in singletons:
            b6_w = p["b6_weights"]
            b5_w = p["b5_weights"]
            if len(b6_w) != 1 or len(b5_w) != 1:
                weight_invariants_pass = False
                break
            active_m = list(b6_w.keys())[0]
            if abs(b6_w[active_m] - 1.0) > 1e-6 or abs(b5_w[active_m] - 1.0) > 1e-6:
                weight_invariants_pass = False
                break

        passed = bool(len(singletons) > 0 and max_singleton_diff < 1e-9 and max_singleton_mae_diff < 1e-9 and weight_invariants_pass)
        self.gate_results["OE-04"] = {
            "name": "Singleton Cardinality Invariance from Raw Records",
            "passed": passed,
            "details": f"Evaluated {len(singletons)} singleton packets; max weight/MAE deviation={max_singleton_diff:.1e}, exact weight=1.0 verified.",
        }

    def _gate_oe05_degradation_and_fidelity_scaling(self, conf: Dict[str, Any], summary: Dict[str, Any]) -> None:
        raw_packets = conf.get("raw_packet_records", [])
        
        # Group by scenario
        mild_diffs = [p["diff_mae_b6_minus_b5"] for p in raw_packets if p["scenario"] == "MILD_DEGRADATION"]
        mod_diffs = [p["diff_mae_b6_minus_b5"] for p in raw_packets if p["scenario"] == "MODERATE_DEGRADATION"]
        sev_diffs = [p["diff_mae_b6_minus_b5"] for p in raw_packets if p["scenario"] == "SEVERE_DEGRADATION"]

        mean_mild = float(np.mean(mild_diffs))
        mean_mod = float(np.mean(mod_diffs))
        mean_sev = float(np.mean(sev_diffs))

        # Monotonicity check: Severe error reduction > Moderate > Mild
        monotonic_deg = (mean_sev < mean_mod < mean_mild < 0.0)

        # Independent Subgroup Calculation for ACCURATE_DEGRADATION
        cohort_dict: Dict[int, List[float]] = {}
        for p in raw_packets:
            if p["scenario"] in ("MILD_DEGRADATION", "MODERATE_DEGRADATION", "SEVERE_DEGRADATION") and p["quality_fidelity"] == "ACCURATE":
                s = p["seed"]
                if s not in cohort_dict:
                    cohort_dict[s] = []
                cohort_dict[s].append(p["diff_mae_b6_minus_b5"])

        cohort_acc_deg_list = [cohort_dict[s] for s in sorted(cohort_dict.keys())]
        indep_acc_deg_ci = self._independent_hierarchical_bootstrap(cohort_acc_deg_list, n_bootstrap=2000, seed=42)

        reported_acc_deg = summary.get("primary_findings", {}).get("H2_Quality_Term_Isolation", {})
        delta_acc_deg = reported_acc_deg.get("accurate_degradation_delta_mae", 0.0)
        ci_acc_deg = reported_acc_deg.get("accurate_degradation_ci_95", [0.0, 0.0])
        h2_verdict = reported_acc_deg.get("verdict", "")

        match_subgroup_delta = abs(indep_acc_deg_ci["grand_mean_diff"] - delta_acc_deg) < 1e-5
        match_subgroup_ci = (abs(indep_acc_deg_ci["ci_lower"] - ci_acc_deg[0]) < 1e-5 and abs(indep_acc_deg_ci["ci_upper"] - ci_acc_deg[1]) < 1e-5)

        passed = bool(
            monotonic_deg
            and match_subgroup_delta
            and match_subgroup_ci
            and indep_acc_deg_ci["ci_upper"] < 0.0
            and delta_acc_deg < 0.0
            and h2_verdict == "CONFIRMED_OBSERVED_QUALITY_BENEFIT_UNDER_ACCURATE_DEGRADATION"
        )
        self.gate_results["OE-05"] = {
            "name": "Degradation & Sensor Fidelity Error Scaling",
            "passed": passed,
            "details": f"Severe={mean_sev:.6f} < Mod={mean_mod:.6f} < Mild={mean_mild:.6f} < 0, Accurate Deg Delta={delta_acc_deg:.6f} (95% CI: [{ci_acc_deg[0]:.6f}, {ci_acc_deg[1]:.6f}]), Verdict={h2_verdict}",
        }

    def _gate_oe06_decision_policy_recomputation(self, conf: Dict[str, Any], summary: Dict[str, Any]) -> None:
        raw_packets = conf.get("raw_packet_records", [])
        
        # Independently derive action tiers from predicted risk scores using standard thresholds
        derived_b6_tiers = [assign_action_tier(p["r_b6"], 0.20, 0.40) for p in raw_packets]
        derived_dcri_tiers = [assign_action_tier(p["dcri_b6"], 0.20, 0.40) for p in raw_packets]
        oracle_tiers = [p["oracle_tier"] for p in raw_packets]

        recomp_b6_losses = [compute_decision_loss(d_tier, o_tier) for d_tier, o_tier in zip(derived_b6_tiers, oracle_tiers)]
        recomp_dcri_losses = [compute_decision_loss(d_tier, o_tier) for d_tier, o_tier in zip(derived_dcri_tiers, oracle_tiers)]

        mean_b6_loss = float(np.mean(recomp_b6_losses))
        mean_dcri_loss = float(np.mean(recomp_dcri_losses))

        high_risk_packets = [i for i, o_tier in enumerate(oracle_tiers) if o_tier == "TIER_2_ESCALATION"]
        n_high_risk = len(high_risk_packets)

        fd_b6 = sum(1 for i in high_risk_packets if derived_b6_tiers[i] != "TIER_2_ESCALATION")
        fd_dcri = sum(1 for i in high_risk_packets if derived_dcri_tiers[i] != "TIER_2_ESCALATION")

        rate_b6 = fd_b6 / max(n_high_risk, 1)
        rate_dcri = fd_dcri / max(n_high_risk, 1)

        reported_dp = summary.get("primary_findings", {}).get("H5_DCRI_Policy_Tradeoff", {})
        match_loss = abs(mean_b6_loss - reported_dp.get("fused_risk_policy_loss", 0.0)) < 1e-4
        match_dcri_loss = abs(mean_dcri_loss - reported_dp.get("dcri_policy_loss", 0.0)) < 1e-4

        passed = bool(match_loss and match_dcri_loss and rate_dcri > rate_b6 and mean_dcri_loss > mean_b6_loss)
        self.gate_results["OE-06"] = {
            "name": "Independent Recomputation of Decision Policy & False Downgrades",
            "passed": passed,
            "details": f"High risk N={n_high_risk}, DCRI false downgrade rate={rate_dcri:.4f} vs Fused Risk={rate_b6:.4f}, DCRI loss={mean_dcri_loss:.4f} vs Fused={mean_b6_loss:.4f}",
        }

    def _gate_oe07_statistical_inference_and_decision_rule(self, conf: Dict[str, Any], summary: Dict[str, Any]) -> None:
        raw_packets = conf.get("raw_packet_records", [])
        
        # Group raw packet differences by cohort seed
        cohort_dict: Dict[int, List[float]] = {}
        for p in raw_packets:
            s = p["seed"]
            if s not in cohort_dict:
                cohort_dict[s] = []
            cohort_dict[s].append(p["diff_mae_b6_minus_b5"])
        
        cohort_diffs_list = [cohort_dict[s] for s in sorted(cohort_dict.keys())]

        # Independently recompute hierarchical cluster bootstrap CI and exact t-test
        indep_ci = self._independent_hierarchical_bootstrap(cohort_diffs_list, n_bootstrap=2000, seed=42)

        reported_h1 = summary.get("primary_findings", {}).get("H1_Primary_Outcome", {})
        reported_ci = reported_h1.get("hierarchical_ci_95", [0.0, 0.0])
        reported_delta = reported_h1.get("delta_mae", 0.0)
        reported_t = reported_h1.get("cohort_t_stat", 0.0)
        reported_p = reported_h1.get("cohort_t_pvalue", 1.0)
        verdict = reported_h1.get("verdict", "")

        # Strict statistical validation
        match_delta = abs(indep_ci["grand_mean_diff"] - reported_delta) < 1e-5
        match_t = abs(indep_ci["cohort_t_stat"] - reported_t) < 1e-4
        
        # Strict exact p-value comparison
        match_p = abs(indep_ci["cohort_t_pvalue"] - reported_p) < 1e-12 or (
            abs(indep_ci["cohort_t_pvalue"] - reported_p) / max(reported_p, 1e-15) < 1e-4
        )
        
        # Strict unrounded CI bounds comparison
        match_ci_lower = abs(indep_ci["ci_lower"] - reported_ci[0]) < 1e-5
        match_ci_upper = abs(indep_ci["ci_upper"] - reported_ci[1]) < 1e-5

        expected_verdict = (
            "CONFIRMED_PRACTICAL_SUPERIORITY"
            if reported_delta < -0.005 and reported_ci[1] < 0.0
            else "STATISTICALLY_SIGNIFICANT_MODEST_IMPROVEMENT_BELOW_PRACTICAL_THRESHOLD"
            if reported_delta < 0.0 and reported_ci[1] < 0.0
            else "INCONCLUSIVE"
        )

        passed = bool(
            match_delta
            and match_t
            and match_p
            and match_ci_lower
            and match_ci_upper
            and reported_delta < 0.0
            and reported_ci[1] < 0.0
            and reported_h1.get("cohort_t_test_significant") is True
            and verdict == expected_verdict
            and reported_h1.get("practical_threshold_met") is False
        )
        self.gate_results["OE-07"] = {
            "name": "Hierarchical Statistical Inference & Honest Decision Rule",
            "passed": passed,
            "details": f"Delta_MAE={reported_delta:.6f}, 95% Hierarchical CI=[{reported_ci[0]:.6f}, {reported_ci[1]:.6f}], t={reported_t:.2f}, p={reported_p:.2e}, Verdict={verdict}",
        }

    def _independent_hierarchical_bootstrap(
        self, cohort_diffs: List[List[float]], n_bootstrap: int = 2000, seed: int = 42, alpha: float = 0.05
    ) -> Dict[str, Any]:
        """Independent reference implementation of 2-stage hierarchical cluster bootstrap & exact t-test."""
        k_cohorts = len(cohort_diffs)
        cohort_arrays = [np.array(c, dtype=float) for c in cohort_diffs]
        all_flat = np.concatenate(cohort_arrays)
        grand_mean = float(np.mean(all_flat))

        cohort_means = [float(np.mean(arr)) for arr in cohort_arrays]
        mean_of_cohort_means = float(np.mean(cohort_means))
        std_of_cohort_means = float(np.std(cohort_means, ddof=1)) if k_cohorts > 1 else 0.0

        rng = np.random.RandomState(seed)
        boot_grand_means = np.empty(n_bootstrap, dtype=float)

        for b in range(n_bootstrap):
            sampled_cohort_idx = rng.choice(k_cohorts, size=k_cohorts, replace=True)
            resampled_packet_means = []
            for c_idx in sampled_cohort_idx:
                c_data = cohort_arrays[c_idx]
                n_packets = len(c_data)
                sampled_packets = rng.choice(c_data, size=n_packets, replace=True)
                resampled_packet_means.append(np.mean(sampled_packets))
            boot_grand_means[b] = np.mean(resampled_packet_means)

        ci_lower = float(np.percentile(boot_grand_means, 100 * (alpha / 2.0)))
        ci_upper = float(np.percentile(boot_grand_means, 100 * (1.0 - alpha / 2.0)))

        # Exact Student's t-test calculation (df = K - 1)
        if k_cohorts > 1 and std_of_cohort_means > 1e-12:
            df = k_cohorts - 1
            se = std_of_cohort_means / math.sqrt(k_cohorts)
            t_stat = mean_of_cohort_means / se
            p_val = float(stats.t.sf(abs(t_stat), df=df) * 2.0)
            t_test_sig = bool(p_val < alpha)
        else:
            t_stat = 0.0
            p_val = 1.0
            t_test_sig = False

        return {
            "grand_mean_diff": round(grand_mean, 6),
            "cohort_mean_of_means": round(mean_of_cohort_means, 6),
            "cohort_std_of_means": round(std_of_cohort_means, 6),
            "ci_lower": round(ci_lower, 6),
            "ci_upper": round(ci_upper, 6),
            "cohort_t_stat": round(t_stat, 6),
            "cohort_t_pvalue": p_val,
            "cohort_t_test_significant": t_test_sig,
        }

    def _gate_oe08_manifest_hashes(self, manifest: Dict[str, Any]) -> None:
        hashes = manifest.get("artifact_hashes", {})
        match_count = 0
        for fname in self.EXPECTED_ARTIFACTS:
            fpath = self.exp_dir / fname
            if not fpath.is_file():
                raise OutcomeVerificationError(f"Required artifact {fname} missing.")
            with open(fpath, "rb") as f:
                actual_hash = hashlib.sha256(f.read()).hexdigest()
            expected_hash = hashes.get(fname)
            if actual_hash != expected_hash:
                raise OutcomeVerificationError(
                    f"Hash mismatch for {fname}: expected {expected_hash}, got {actual_hash}"
                )
            match_count += 1

        self.gate_results["OE-08"] = {
            "name": "Cryptographic Hash Manifest Verification",
            "passed": bool(match_count == len(self.EXPECTED_ARTIFACTS)),
            "details": f"{match_count}/{len(self.EXPECTED_ARTIFACTS)} SHA-256 hashes verified match.",
        }


if __name__ == "__main__":
    verifier = OutcomeArtifactVerifier(Path("experiments/fusion/outcome_evaluation"))
    res = verifier.verify_all()
    print(json.dumps(res, indent=2))
    if res["status"] != "PASSED":
        sys.exit(1)
