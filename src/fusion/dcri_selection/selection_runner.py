"""
FusionMedAI - Phase C11.13: DCRI Global Uncertainty Penalty Selection Runner
Orchestrates execution of the delta selection experiment, artifact generation,
and cryptographic freeze manifest creation.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import hashlib
import numpy as np

from src.fusion.dcri.cohort_loader import load_frozen_cohort
from src.fusion.baselines.decision_packet import ControlledDecisionPacket
from .selection_config import (
    ALPHA_REF,
    BETA_REF,
    GAMMA_REF,
    ETA_REF,
    DELTA_PROVISIONAL_HISTORICAL,
    SEED,
    N_PACKETS,
    N_BOOTSTRAPS,
    CANDIDATE_DELTA_GRID,
    ACTIVE_REGIMES,
    ALL_REGIMES,
)
from .candidate_grid import (
    DeltaCandidateItem,
    get_candidate_grid,
    validate_candidate_grid,
)
from .selection_evaluator import SelectionEvaluator
from .regime_evaluator import RegimeEvaluator
from .paired_bootstrap import PairedBootstrapComparator
from .selection_decision import SelectionDecisionEngine


def compute_sha256_file(filepath: Path) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


class DeltaSelectionExperimentRunner:
    """
    Executes the complete Phase C11.13 DCRI uncertainty penalty selection protocol.
    """

    def __init__(
        self,
        repo_root: Optional[Path] = None,
        seed: int = SEED,
        n_bootstraps: int = N_BOOTSTRAPS,
    ):
        self.repo_root = repo_root or Path(__file__).resolve().parents[3]
        self.seed = seed
        self.n_bootstraps = n_bootstraps
        self.grid = get_candidate_grid()
        validate_candidate_grid(self.grid)
        
        self.evaluator = SelectionEvaluator()
        self.regime_evaluator = RegimeEvaluator()
        self.bootstrap_comparator = PairedBootstrapComparator(n_bootstraps=self.n_bootstraps, seed=self.seed)
        self.decision_engine = SelectionDecisionEngine()

    def load_cohort(self) -> List[ControlledDecisionPacket]:
        """Loads the frozen cohort of N=500 ControlledDecisionPackets."""
        return load_frozen_cohort(self.repo_root, n_packets=N_PACKETS, seed=self.seed)

    def run_full_selection(self) -> Dict[str, Any]:
        """
        Executes all evaluation steps and returns in-memory results dictionary.
        """
        cohort = self.load_cohort()
        
        # 1. Precompute base states
        base_states = self.evaluator.precompute_packet_base_state(cohort)
        
        # 2. Evaluate all candidate delta items
        candidate_evals: List[Dict[str, Any]] = []
        for cand in self.grid:
            cand_res = self.evaluator.evaluate_candidate_delta(cand, base_states)
            candidate_evals.append(cand_res)
            
        # 3. Regime analysis
        regime_res = self.regime_evaluator.evaluate_regimes(candidate_evals, cohort)
        
        # 4. Bootstrap comparisons
        boot_res = self.bootstrap_comparator.run_comparisons(candidate_evals, base_states)
        
        # 5. Selection hierarchy execution
        selection_res = self.decision_engine.apply_selection_hierarchy(
            candidate_evals=candidate_evals,
            regime_results=regime_res,
            bootstrap_results=boot_res,
        )
        
        return {
            "grid": self.grid,
            "base_states": base_states,
            "candidate_evals": candidate_evals,
            "regime_results": regime_res,
            "bootstrap_results": boot_res,
            "selection_results": selection_res,
        }

    def run_and_save_artifacts(self, output_dir: Optional[Path] = None) -> Dict[str, Path]:
        """
        Executes full selection, exports all 8 JSON artifacts, and generates SHA-256 freeze manifest.
        
        Returns:
            Dictionary mapping artifact names to their file paths.
        """
        out_dir = output_dir or (self.repo_root / "experiments" / "fusion" / "dcri_selection" / "results")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        results = self.run_full_selection()
        candidate_evals = results["candidate_evals"]
        regime_res = results["regime_results"]
        boot_res = results["bootstrap_results"]
        sel_res = results["selection_results"]
        
        # 1. delta_selection_config.json
        cfg_data = {
            "phase": "C11.13",
            "title": "DCRI Global Uncertainty Penalty Selection Protocol",
            "frozen_router_coefficients": {
                "alpha": ALPHA_REF,
                "beta": BETA_REF,
                "gamma": GAMMA_REF,
                "eta": ETA_REF,
            },
            "candidate_delta_grid": list(CANDIDATE_DELTA_GRID),
            "historical_provisional_delta": DELTA_PROVISIONAL_HISTORICAL,
            "cohort_size": N_PACKETS,
            "seed": self.seed,
            "n_bootstraps": self.n_bootstraps,
            "regimes_evaluated": list(ALL_REGIMES),
            "hard_validity_invariants": [
                "Strict Determinism",
                "Monotonic Penalty Invariant: DCRI_delta2 <= DCRI_delta1 for delta2 > delta1",
                "Zero-Penalty Identity: DCRI_0 == R_fusion exactly",
                "Analytical Derivative Consistency: dDCRI/ddelta == -U_sum",
                "Unclamped Negative Domain Invariant",
                "No Upstream Parameter Changes or Label Leakage",
            ],
        }
        cfg_path = out_dir / "delta_selection_config.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_data, f, indent=2)
            
        # 2. candidate_grid.json
        grid_data = {
            "total_candidates": len(self.grid),
            "candidates": [item.to_dict() for item in self.grid],
        }
        grid_path = out_dir / "candidate_grid.json"
        with open(grid_path, "w", encoding="utf-8") as f:
            json.dump(grid_data, f, indent=2)
            
        # 3. delta_results.json (Aggregated candidate metrics without full packet dumps for compactness)
        delta_summary = []
        for c in candidate_evals:
            c_copy = {k: v for k, v in c.items() if k != "packet_evaluations"}
            delta_summary.append(c_copy)
        delta_res_path = out_dir / "delta_results.json"
        with open(delta_res_path, "w", encoding="utf-8") as f:
            json.dump({"candidates": delta_summary}, f, indent=2)
            
        # 4. regime_results.json
        regime_path = out_dir / "regime_results.json"
        with open(regime_path, "w", encoding="utf-8") as f:
            json.dump(regime_res, f, indent=2)
            
        # 5. distribution_results.json (Detailed quantiles, IQR, exceedance rates)
        dist_data = {
            "distribution_profiles": [
                {
                    "candidate_id": c["candidate_id"],
                    "delta": c["delta"],
                    "mean": c["mean_dcri"],
                    "median": c["median_dcri"],
                    "std": c["std_dcri"],
                    "min": c["min_dcri"],
                    "max": c["max_dcri"],
                    "quantiles": c["quantiles"],
                    "negative_rate": c["negative_rate"],
                    "penalty_to_base_ratio": c["penalty_to_base_ratio"],
                    "penalty_exceedance_rates": c["penalty_exceedance_rates"],
                }
                for c in candidate_evals
            ]
        }
        dist_path = out_dir / "distribution_results.json"
        with open(dist_path, "w", encoding="utf-8") as f:
            json.dump(dist_data, f, indent=2)
            
        # 6. rank_stability.json
        rank_data = {
            "rank_stabilities": [
                {
                    "candidate_id": c["candidate_id"],
                    "delta": c["delta"],
                    "spearman_rho": c["rank_stability"]["spearman_rho"],
                    "spearman_pvalue": c["rank_stability"]["spearman_pvalue"],
                    "kendall_tau": c["rank_stability"]["kendall_tau"],
                    "kendall_pvalue": c["rank_stability"]["kendall_pvalue"],
                }
                for c in candidate_evals
            ]
        }
        rank_path = out_dir / "rank_stability.json"
        with open(rank_path, "w", encoding="utf-8") as f:
            json.dump(rank_data, f, indent=2)
            
        # 7. bootstrap_comparisons.json
        boot_path = out_dir / "bootstrap_comparisons.json"
        with open(boot_path, "w", encoding="utf-8") as f:
            json.dump(boot_res, f, indent=2)
            
        # 8. selection_summary.json
        sel_path = out_dir / "selection_summary.json"
        with open(sel_path, "w", encoding="utf-8") as f:
            json.dump(sel_res, f, indent=2)
            
        # 9. freeze_manifest.json (Cryptographic SHA-256 hashes of all artifacts)
        artifact_files = [
            cfg_path,
            grid_path,
            delta_res_path,
            regime_path,
            dist_path,
            rank_path,
            boot_path,
            sel_path,
        ]
        
        from datetime import datetime, timezone
        manifest_data = {
            "phase": "C11.13",
            "freeze_scope": "DCRI Uncertainty Penalty Parameter Selection",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "seed": self.seed,
            "cohort_size": N_PACKETS,
            "selected_delta": sel_res["selected_delta"],
            "selected_candidate_id": sel_res["selected_candidate_id"],
            "file_hashes": {p.name: compute_sha256_file(p) for p in artifact_files},
        }
        manifest_path = out_dir / "freeze_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
            
        return {
            "config": cfg_path,
            "grid": grid_path,
            "delta_results": delta_res_path,
            "regime_results": regime_path,
            "distribution_results": dist_path,
            "rank_stability": rank_path,
            "bootstrap_comparisons": boot_path,
            "selection_summary": sel_path,
            "freeze_manifest": manifest_path,
        }
