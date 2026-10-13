"""
FusionMedAI - Phase C11.14: DCRI Decision Policy Analysis Runner
Orchestrates execution of the decision policy analysis, exports 6 JSON artifacts,
and generates cryptographic SHA-256 freeze manifest.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import hashlib
import numpy as np

from src.fusion.dcri.cohort_loader import load_frozen_cohort
from src.fusion.baselines.decision_packet import ControlledDecisionPacket
from .policy_config import (
    ALPHA_REF,
    BETA_REF,
    GAMMA_REF,
    ETA_REF,
    DELTA_FROZEN,
    DELTA_PROVISIONAL_HISTORICAL,
    SEED,
    N_PACKETS,
    N_BOOTSTRAPS,
    TAU_1_DEFAULT,
    TAU_2_DEFAULT,
    TAU_1_GRID,
    TAU_2_GRID,
    ACTIVE_REGIMES,
    ALL_REGIMES,
    ACTION_NAMES,
    ACTION_KEYS,
)
from .policy_engine import PolicyEvaluator
from .threshold_sweeper import ThresholdSensitivitySweeper
from .regime_policy_evaluator import RegimePolicyEvaluator
from .robustness_evaluator import RobustnessEvaluator
from .policy_stats import PolicyStatisticalAnalyzer


def compute_sha256_file(filepath: Path) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


class PolicyAnalysisExperimentRunner:
    """
    Executes Phase C11.14 Decision Policy Sensitivity and Operating-Behavior Analysis.
    
    Cohort Paths:
        - Authoritative Canonical Redesigned Cohort: Default manifest path
          `experiments/fusion/canonical_redesigned_rerun/01_canonical_redesigned_packet_manifest.json`
          (SHA-256 cf877d74e306...). Loaded by default via `cohort_loader.load_frozen_cohort()`.
          Evaluated via `run_canonical_safeguard()` or when manifest_path is default.
        - Historical Baseline Decision Cohort: Manifest path `experiments/fusion/dcri/packet_manifest.json`
          (SHA-256 1ff37fbeeba3...). Preserved for historical comparison; requires explicit opt-in.
    """

    def __init__(
        self,
        repo_root: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        seed: int = SEED,
        n_bootstraps: int = N_BOOTSTRAPS,
        manifest_path: Optional[Path] = None,
    ):
        self.repo_root = repo_root or Path(__file__).resolve().parents[3]
        self.seed = seed
        self.n_bootstraps = n_bootstraps
        self.output_dir = output_dir or (self.repo_root / "experiments" / "fusion" / "dcri_policy_analysis" / "results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = manifest_path

    def run_all(self) -> Dict[str, Any]:
        """
        Executes the entire Phase C11.14 experiment pipeline on the baseline decision cohort.
        """
        # 1. Load frozen cohort

        cohort = load_frozen_cohort(
            repo_root=self.repo_root,
            n_packets=N_PACKETS,
            seed=self.seed,
            manifest_path=self.manifest_path,
        )

        # 2. Base state evaluation
        evaluator = PolicyEvaluator(delta=DELTA_FROZEN)
        base_states = evaluator.precompute_packet_base_state(cohort)

        # 3. Primary Policy Evaluation (Nominal Tau_1=0.20, Tau_2=0.40)
        primary_eval = evaluator.evaluate_policies(
            base_states=base_states,
            tau_1=TAU_1_DEFAULT,
            tau_2=TAU_2_DEFAULT,
        )

        # 4. Threshold Sensitivity Sweep
        sweeper = ThresholdSensitivitySweeper(
            tau_1_grid=TAU_1_GRID,
            tau_2_grid=TAU_2_GRID,
            delta=DELTA_FROZEN,
        )
        threshold_sweep_results = sweeper.sweep(base_states)

        # 5. Regime Stratification
        regime_evaluator = RegimePolicyEvaluator(
            delta=DELTA_FROZEN,
            tau_1=TAU_1_DEFAULT,
            tau_2=TAU_2_DEFAULT,
        )
        regime_results = regime_evaluator.evaluate_regimes(cohort)

        # 6. Robustness Evaluations
        robustness_evaluator = RobustnessEvaluator(
            delta=DELTA_FROZEN,
            tau_1=TAU_1_DEFAULT,
            tau_2=TAU_2_DEFAULT,
        )
        unc_scaling_res = robustness_evaluator.evaluate_uncertainty_scaling(base_states)
        jitter_res = robustness_evaluator.evaluate_threshold_jitter(base_states)

        # 7. Statistical Analysis (Paired Bootstrap B=1000)
        stats_analyzer = PolicyStatisticalAnalyzer(
            n_bootstraps=self.n_bootstraps,
            seed=self.seed,
        )
        stat_results = stats_analyzer.compute_paired_statistics(
            base_states=base_states,
            per_packet_details=primary_eval.per_packet_details,
        )

        # 7.5 Safeguarded Policy Evaluation
        safeguarded_eval = evaluator.evaluate_safeguarded_policy(
            base_states=base_states,
            tau_1=TAU_1_DEFAULT,
            tau_2=TAU_2_DEFAULT,
        )

        safeguarded_artifact = {
            "phase": "C11.14",
            "experiment_title": "DCRI Decision Policy Safeguard: Ambiguity Gateway and Repeat-Exam Policy",
            "safeguarded_evaluation": safeguarded_eval.to_dict(),
        }

        # 8. Construct Output Artifacts Dictionaries
        config_artifact = {
            "phase": "C11.14",
            "experiment_title": "Decision-Policy Sensitivity and Operating-Behavior Analysis of Frozen DCRI Index",
            "frozen_parameters": {
                "delta": DELTA_FROZEN,
                "delta_provisional_historical": DELTA_PROVISIONAL_HISTORICAL,
                "router_coefficients": {
                    "alpha": ALPHA_REF,
                    "beta": BETA_REF,
                    "gamma": GAMMA_REF,
                    "eta": ETA_REF,
                },
                "cohort_size": N_PACKETS,
                "seed": self.seed,
                "n_bootstraps": self.n_bootstraps,
            },
            "policy_definitions": {
                "action_categories": ACTION_NAMES,
                "action_keys": ACTION_KEYS,
                "standard_thresholds": {
                    "tau_1": TAU_1_DEFAULT,
                    "tau_2": TAU_2_DEFAULT,
                },
                "threshold_sweep_grids": {
                    "tau_1_grid": list(TAU_1_GRID),
                    "tau_2_grid": list(TAU_2_GRID),
                },
            },
            "regimes_evaluated": list(ALL_REGIMES),
            "status": "FROZEN",
        }

        threshold_artifact = {
            "phase": "C11.14",
            "primary_operating_point": primary_eval.to_dict(),
            "threshold_sweep_results": threshold_sweep_results,
            "summary_insights": {
                "nominal_reclassification_rate": primary_eval.reclassification_rate,
                "nominal_downgraded_rate": primary_eval.downgraded_rate,
                "nominal_upgraded_rate": primary_eval.upgraded_rate,
                "nominal_escalation_reduction_rate": primary_eval.escalation_reduction_rate,
                "monotonic_downgrade_invariant": (primary_eval.upgraded_count == 0),
            },
        }

        regime_artifact = {
            "phase": "C11.14",
            "delta": DELTA_FROZEN,
            "standard_thresholds": {"tau_1": TAU_1_DEFAULT, "tau_2": TAU_2_DEFAULT},
            "regime_results": regime_results,
        }

        robustness_artifact = {
            "phase": "C11.14",
            "uncertainty_scaling_perturbation": unc_scaling_res,
            "threshold_jitter_perturbation": jitter_res,
        }

        statistical_artifact = {
            "phase": "C11.14",
            "statistical_summary": stat_results,
        }

        # 9. Save JSON Artifacts to Disk
        artifact_files: Dict[str, Path] = {
            "policy_config.json": self.output_dir / "policy_config.json",
            "threshold_results.json": self.output_dir / "threshold_results.json",
            "regime_results.json": self.output_dir / "regime_results.json",
            "robustness_results.json": self.output_dir / "robustness_results.json",
            "statistical_results.json": self.output_dir / "statistical_results.json",
            "safeguarded_policy_results.json": self.output_dir / "safeguarded_policy_results.json",
        }

        with open(artifact_files["policy_config.json"], "w", encoding="utf-8") as f:
            json.dump(config_artifact, f, indent=2)

        with open(artifact_files["threshold_results.json"], "w", encoding="utf-8") as f:
            json.dump(threshold_artifact, f, indent=2)

        with open(artifact_files["regime_results.json"], "w", encoding="utf-8") as f:
            json.dump(regime_artifact, f, indent=2)

        with open(artifact_files["robustness_results.json"], "w", encoding="utf-8") as f:
            json.dump(robustness_artifact, f, indent=2)

        with open(artifact_files["statistical_results.json"], "w", encoding="utf-8") as f:
            json.dump(statistical_artifact, f, indent=2)

        with open(artifact_files["safeguarded_policy_results.json"], "w", encoding="utf-8") as f:
            json.dump(safeguarded_artifact, f, indent=2)

        # 10. Generate Freeze Manifest with SHA-256 Hashes
        manifest: Dict[str, Any] = {
            "phase": "C11.14",
            "manifest_version": "1.0.0",
            "cohort_seed": self.seed,
            "n_packets": N_PACKETS,
            "delta_frozen": DELTA_FROZEN,
            "artifacts": {},
        }

        for filename, filepath in artifact_files.items():
            manifest["artifacts"][filename] = {
                "sha256": compute_sha256_file(filepath),
                "size_bytes": filepath.stat().st_size,
            }

        manifest_path = self.output_dir / "freeze_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return {
            "manifest": manifest,
            "output_dir": str(self.output_dir),
            "primary_eval": primary_eval.to_dict(),
            "safeguarded_eval": safeguarded_eval.to_dict(),
        }

    def run_canonical_safeguard(self) -> Dict[str, Any]:
        """
        Executes Policy S Safeguard evaluation on the Canonical Redesigned Cohort
        (01_canonical_redesigned_packet_manifest.json), exporting canonical_safeguarded_policy_results.json.
        """
        canonical_manifest_path = (
            self.repo_root / "experiments" / "fusion" / "canonical_redesigned_rerun" / "01_canonical_redesigned_packet_manifest.json"
        )
        cohort = load_frozen_cohort(
            repo_root=self.repo_root,
            n_packets=N_PACKETS,
            seed=self.seed,
            manifest_path=canonical_manifest_path,
        )
        evaluator = PolicyEvaluator(delta=DELTA_FROZEN)
        base_states = evaluator.precompute_packet_base_state(cohort)
        safeguarded_eval = evaluator.evaluate_safeguarded_policy(
            base_states=base_states,
            tau_1=TAU_1_DEFAULT,
            tau_2=TAU_2_DEFAULT,
        )
        canonical_artifact = {
            "phase": "C11_CANONICAL_RERUN",
            "cohort_source": "01_canonical_redesigned_packet_manifest.json",
            "manifest_sha256": compute_sha256_file(canonical_manifest_path),
            "experiment_title": "Canonical Redesigned Cohort Policy S Safeguarded Evaluation",
            "safeguarded_evaluation": safeguarded_eval.to_dict(),
        }
        out_path = self.repo_root / "experiments" / "fusion" / "canonical_redesigned_rerun" / "canonical_safeguarded_policy_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(canonical_artifact, f, indent=2)
        return canonical_artifact


if __name__ == "__main__":
    runner = PolicyAnalysisExperimentRunner()
    result = runner.run_all()
    print(f"Phase C11.14 execution complete. Manifest created with {len(result['manifest']['artifacts'])} artifacts.")
    canonical_res = runner.run_canonical_safeguard()
    print(f"Canonical Redesigned Cohort Policy S evaluation complete. Exported canonical_safeguarded_policy_results.json.")
