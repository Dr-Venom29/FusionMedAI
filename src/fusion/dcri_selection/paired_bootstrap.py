"""
FusionMedAI - Phase C11.13: Paired Bootstrap Delta Comparison Engine
Executes B=1000 paired bootstrap resamples to compute non-parametric confidence intervals
for DCRI differences between candidate delta conditions.
"""

from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import numpy as np

from .selection_config import N_BOOTSTRAPS, SEED, BOOTSTRAP_CI_ALPHA


@dataclass(frozen=True)
class BootstrapComparisonResult:
    """Represents paired bootstrap difference statistics between two delta conditions."""
    candidate_a: str
    delta_a: float
    candidate_b: str
    delta_b: float
    delta_diff: float
    observed_diff: float
    bootstrap_mean_diff: float
    std_err: float
    ci_lower: float
    ci_upper: float
    n_bootstraps: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_a": self.candidate_a,
            "delta_a": self.delta_a,
            "candidate_b": self.candidate_b,
            "delta_b": self.delta_b,
            "delta_diff": self.delta_diff,
            "observed_diff": self.observed_diff,
            "bootstrap_mean_diff": self.bootstrap_mean_diff,
            "std_err": self.std_err,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "n_bootstraps": self.n_bootstraps,
        }


class PairedBootstrapComparator:
    """
    Computes deterministic paired bootstrap confidence intervals for candidate delta comparisons.
    """

    def __init__(self, n_bootstraps: int = N_BOOTSTRAPS, seed: int = SEED):
        self.n_bootstraps = n_bootstraps
        self.seed = seed

    def run_comparisons(
        self,
        candidate_evals: List[Dict[str, Any]],
        base_states: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Runs paired bootstrap for key comparisons.
        
        Comparisons evaluated:
        1. Baseline vs each candidate: delta=0.0 vs delta in {0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00}
        2. Adjacent step comparisons: (0 vs .05), (.05 vs .10), (.10 vs .15), (.15 vs .20), (.20 vs .25), (.25 vs .30), (.30 vs .40), (.40 vs .50), (.50 vs .75), (.75 vs 1.00)
        3. Historical reference comparisons: delta=0.20 vs delta in {0.00, 0.05, 0.10, 0.15, 0.25, 0.30, 0.50}
        """
        active_states = [s for s in base_states if not s["is_empty"]]
        n_samples = len(active_states)
        
        # Verify strict packet-ID alignment across all candidates
        ref_pids = [p["packet_id"] for p in candidate_evals[0]["packet_evaluations"]]
        for cand in candidate_evals:
            cand_pids = [p["packet_id"] for p in cand["packet_evaluations"]]
            if cand_pids != ref_pids:
                raise ValueError(
                    f"Packet ID ordering mismatch in candidate {cand['candidate_id']}: "
                    f"expected {len(ref_pids)} packets matching reference order."
                )
        
        # Precompute per-packet DCRI arrays for each candidate
        cand_map: Dict[str, Dict[str, Any]] = {c["candidate_id"]: c for c in candidate_evals}
        packet_dcri_by_cand: Dict[str, np.ndarray] = {}
        
        for cid, cand in cand_map.items():
            arr = np.array([p["dcri"] for p in cand["packet_evaluations"]], dtype=np.float64)
            packet_dcri_by_cand[cid] = arr
            
        # Deterministic bootstrap resample matrix (B x N)
        rng = np.random.RandomState(self.seed)
        resample_indices = rng.randint(0, n_samples, size=(self.n_bootstraps, n_samples))
        
        # Function to compute paired comparison
        def _compare_pair(cid_a: str, cid_b: str) -> BootstrapComparisonResult:
            arr_a = packet_dcri_by_cand[cid_a]
            arr_b = packet_dcri_by_cand[cid_b]
            delta_a = cand_map[cid_a]["delta"]
            delta_b = cand_map[cid_b]["delta"]
            
            # Original observed sample paired difference: arr_a - arr_b
            # Since DCRI_a - DCRI_b = -(delta_a - delta_b) * U_sum
            diffs_per_packet = arr_a - arr_b
            observed_diff = float(np.mean(diffs_per_packet))
            
            # Bootstrap distribution over sample replicates
            boot_means = np.mean(diffs_per_packet[resample_indices], axis=1)
            bootstrap_mean_diff = float(np.mean(boot_means))
            std_err = float(np.std(boot_means, ddof=1))
            
            alpha_lower = 100.0 * (BOOTSTRAP_CI_ALPHA / 2.0)
            alpha_upper = 100.0 * (1.0 - BOOTSTRAP_CI_ALPHA / 2.0)
            ci_lower = float(np.percentile(boot_means, alpha_lower))
            ci_upper = float(np.percentile(boot_means, alpha_upper))
            
            return BootstrapComparisonResult(
                candidate_a=cid_a,
                delta_a=delta_a,
                candidate_b=cid_b,
                delta_b=delta_b,
                delta_diff=float(delta_a - delta_b),
                observed_diff=observed_diff,
                bootstrap_mean_diff=bootstrap_mean_diff,
                std_err=std_err,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                n_bootstraps=self.n_bootstraps,
            )
            
        # 1. Zero vs all
        zero_cid = "D00"
        zero_vs_all = []
        for cand in candidate_evals:
            cid = cand["candidate_id"]
            if cid != zero_cid:
                zero_vs_all.append(_compare_pair(zero_cid, cid).to_dict())
                
        # 2. Adjacent steps
        sorted_cids = [c["candidate_id"] for c in candidate_evals]
        adjacent_steps = []
        for i in range(len(sorted_cids) - 1):
            cid_a = sorted_cids[i]
            cid_b = sorted_cids[i+1]
            adjacent_steps.append(_compare_pair(cid_a, cid_b).to_dict())
            
        # 3. Provisional 0.20 vs others
        prov_cid = "D20"
        prov_comparisons = []
        target_prov_cids = ["D00", "D05", "D10", "D15", "D25", "D30", "D50", "D100"]
        for cid in target_prov_cids:
            if cid in cand_map and cid != prov_cid:
                prov_comparisons.append(_compare_pair(prov_cid, cid).to_dict())
                
        return {
            "n_bootstraps": self.n_bootstraps,
            "seed": self.seed,
            "zero_vs_all": zero_vs_all,
            "adjacent_steps": adjacent_steps,
            "provisional_vs_others": prov_comparisons,
        }
