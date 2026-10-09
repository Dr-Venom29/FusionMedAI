"""
FusionMedAI - Phase C11.13: Pre-Specified Selection Decision Logic
Implements the generic multi-tiered selection hierarchy to determine the frozen delta* parameter.
"""

from typing import Dict, List, Tuple, Any, Optional
import math
import numpy as np

from .selection_config import (
    MAX_TOLERABLE_NEGATIVE_RATE,
    MAX_TOLERABLE_MEAN_PENALTY_RATIO,
    MIN_RANK_STABILITY_SPEARMAN,
    MIN_MEANINGFUL_PENALTY_RATIO,
    MAX_PREFERRED_NEGATIVE_RATE,
    MIN_PREFERRED_RANK_STABILITY,
    MAX_ACCEPTABLE_FLOAT_TOLERANCE,
    RESIDUAL_ERROR_TOLERANCE,
    ACTIVE_REGIMES,
)


class SelectionDecisionEngine:
    """
    Executes the pre-specified 4-step selection hierarchy to determine the selected delta*.
    """

    def apply_selection_hierarchy(
        self,
        candidate_evals: List[Dict[str, Any]],
        regime_results: Dict[str, Any],
        bootstrap_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Applies pre-specified decision rules to evaluate and select delta*.
        
        Hierarchy Steps:
        Step 1 - Hard Validity Filter: Reject candidates violating residual tolerance, zero-penalty identity, monotonicity, or numerical finiteness.
        Step 2 - Behavioral Feasibility Filter: Reject candidates exceeding negative-rate (>35%), penalty ratio (>60%), rank distortion (rho < 0.90), or regime negative collapse (>45%).
        Step 3 - Preferred Criteria Evaluation: Identify candidates delivering meaningful uncertainty attenuation (penalty ratio >= 20%) with negative rate <= 10%, rank stability rho >= 0.98, and bootstrap-verified positive discount (CI lower > 0).
        Step 4 - Parsimony Decision Rule: Select the smallest delta among preferred candidates; fallback to delta=0.00 if no non-zero candidate meets preferred criteria.
        """
        audit_trail: List[Dict[str, Any]] = []
        valid_candidates: List[Dict[str, Any]] = []
        
        # Precompute candidate map for relative monotonicity checking
        sorted_evals = sorted(candidate_evals, key=lambda c: c["delta"])
        
        # Step 1: Hard Validity Checks
        for idx, cand in enumerate(sorted_evals):
            cid = cand["candidate_id"]
            delta = cand["delta"]
            res_err = cand["max_residual_error"]
            reasons = []
            
            # Check 1A: Floating point residual tolerance
            if res_err > RESIDUAL_ERROR_TOLERANCE:
                reasons.append(f"Residual error {res_err:.2e} exceeds float tolerance {RESIDUAL_ERROR_TOLERANCE:.2e}")
                
            # Check 1B: Finite metrics
            if not np.isfinite(cand["mean_dcri"]) or not np.isfinite(cand["std_dcri"]):
                reasons.append("Non-finite (NaN or Inf) detected in DCRI statistics")
                
            # Check 1C: Zero-penalty identity at delta=0
            if cand["is_zero"]:
                max_d0_diff = max(abs(p["dcri"] - p["r_fusion"]) for p in cand["packet_evaluations"])
                if max_d0_diff > 1e-14:
                    reasons.append(f"Zero-penalty identity violated: max diff {max_d0_diff:.2e} > 1e-14")
                    
            # Check 1D: Packet-level monotonic decay relative to lower delta
            if idx > 0:
                prev_cand = sorted_evals[idx - 1]
                for p_curr, p_prev in zip(cand["packet_evaluations"], prev_cand["packet_evaluations"]):
                    if p_curr["dcri"] > p_prev["dcri"] + 1e-14:
                        reasons.append(f"Packet {p_curr['packet_id']} violates monotonic decay ({p_curr['dcri']} > {p_prev['dcri']})")
                        break
                        
            is_valid = len(reasons) == 0
            audit_trail.append({
                "candidate_id": cid,
                "delta": delta,
                "step": 1,
                "stage": "hard_validity",
                "passed": is_valid,
                "reasons": reasons,
            })
            if is_valid:
                valid_candidates.append(cand)
                
        # Step 2: Behavioral Feasibility Checks
        feasible_candidates: List[Dict[str, Any]] = []
        reg_by_cand = regime_results.get("by_candidate", {})
        
        for cand in valid_candidates:
            cid = cand["candidate_id"]
            delta = cand["delta"]
            neg_rate = cand["negative_rate"]
            pen_ratio = cand["penalty_to_base_ratio"]
            spearman_rho = cand["rank_stability"]["spearman_rho"]
            reasons = []
            
            # Constraint 2A: Cohort Negative DCRI rate
            if neg_rate > MAX_TOLERABLE_NEGATIVE_RATE:
                reasons.append(f"Cohort negative DCRI rate {neg_rate:.1%} exceeds threshold {MAX_TOLERABLE_NEGATIVE_RATE:.1%}")
                
            # Constraint 2B: Mean penalty ratio
            if pen_ratio > MAX_TOLERABLE_MEAN_PENALTY_RATIO:
                reasons.append(f"Penalty-to-base ratio {pen_ratio:.1%} exceeds threshold {MAX_TOLERABLE_MEAN_PENALTY_RATIO:.1%}")
                
            # Constraint 2C: Spearman rank stability floor
            if spearman_rho < MIN_RANK_STABILITY_SPEARMAN:
                reasons.append(f"Spearman rank stability {spearman_rho:.4f} is below minimum {MIN_RANK_STABILITY_SPEARMAN:.2f}")
                
            # Constraint 2D: Regime sanity across active availability states
            if cid in reg_by_cand:
                cand_regs = reg_by_cand[cid]["regimes"]
                for reg_name in ACTIVE_REGIMES:
                    reg_data = cand_regs.get(reg_name, {})
                    reg_neg_rate = reg_data.get("negative_rate", 0.0)
                    if reg_neg_rate > 0.45:  # No active regime should experience >45% negative rate
                        reasons.append(f"Regime {reg_name} negative rate {reg_neg_rate:.1%} exceeds regime threshold 45.0%")
                        break
                        
            is_feasible = len(reasons) == 0
            audit_trail.append({
                "candidate_id": cid,
                "delta": delta,
                "step": 2,
                "stage": "behavioral_feasibility",
                "passed": is_feasible,
                "reasons": reasons,
            })
            if is_feasible:
                feasible_candidates.append(cand)
                
        # Step 3 & 4: Parsimonious Decision Selection Hierarchy
        if not feasible_candidates:
            return {
                "selected_delta": None,
                "selected_candidate_id": "NO_FEASIBLE_CANDIDATE",
                "selected_candidate_name": "No Feasible Candidate",
                "status": "NO_FEASIBLE_CANDIDATE",
                "feasible_candidate_ids": [],
                "rejected_candidate_ids": [c["candidate_id"] for c in candidate_evals],
                "audit_trail": audit_trail,
                "selection_rationale": ["No candidate passed all Tier 1 validity and Tier 2 feasibility constraints."],
                "selected_metrics": {},
            }

        # Build lookup for bootstrap zero vs candidate CI lower bounds
        boot_zero_map: Dict[str, float] = {}
        for b_comp in bootstrap_results.get("zero_vs_all", []):
            boot_zero_map[b_comp["candidate_b"]] = b_comp["ci_lower"]

        # Identify non-zero candidates meeting preferred criteria
        preferred_candidates = []
        for c in feasible_candidates:
            if c["delta"] > 0.0:
                cid = c["candidate_id"]
                meets_pen_ratio = (c["penalty_to_base_ratio"] >= MIN_MEANINGFUL_PENALTY_RATIO)
                meets_neg_rate = (c["negative_rate"] <= MAX_PREFERRED_NEGATIVE_RATE)
                meets_rank = (c["rank_stability"]["spearman_rho"] >= MIN_PREFERRED_RANK_STABILITY)
                # Bootstrap confidence interval must confirm strictly positive uncertainty discount (ci_lower > 0)
                ci_lower = boot_zero_map.get(cid, 0.0)
                meets_boot = (ci_lower > 0.0)
                
                if meets_pen_ratio and meets_neg_rate and meets_rank and meets_boot:
                    preferred_candidates.append(c)

        if preferred_candidates:
            # Parsimony principle: select the smallest delta among preferred candidates
            selected_candidate = min(preferred_candidates, key=lambda c: c["delta"])
            selection_rationale = [
                "1. Passed all Tier 1 Hard Validity Invariants (zero residual error, zero-identity, strict monotonic decay).",
                f"2. Passed Tier 2 Behavioral Feasibility constraints (cohort negative rate <= {MAX_TOLERABLE_NEGATIVE_RATE:.0%}, penalty ratio <= {MAX_TOLERABLE_MEAN_PENALTY_RATIO:.0%}, rho >= {MIN_RANK_STABILITY_SPEARMAN:.2f}, active regime negative rates <= 45%).",
                f"3. Satisfies Tier 3/4 Preferred Parsimony Criteria: delivers meaningful uncertainty discounting (penalty ratio {selected_candidate['penalty_to_base_ratio']:.1%} >= {MIN_MEANINGFUL_PENALTY_RATIO:.0%}) with negative rate containment ({selected_candidate['negative_rate']:.1%} <= {MAX_PREFERRED_NEGATIVE_RATE:.0%}), high rank fidelity (Spearman rho = {selected_candidate['rank_stability']['spearman_rho']:.4f} >= {MIN_PREFERRED_RANK_STABILITY:.2f}), and 95% bootstrap CI confirming strictly positive uncertainty discount.",
                f"4. Selected as the smallest non-zero delta ({selected_candidate['delta']:.2f}) meeting all preferred criteria under parsimony rule.",
            ]
        else:
            # Fallback to delta=0.00 if feasible
            zero_cand = next((c for c in feasible_candidates if c["delta"] == 0.0), None)
            if zero_cand is not None:
                selected_candidate = zero_cand
                selection_rationale = [
                    "1. No non-zero candidate satisfied preferred uncertainty discounting and negative-rate containment criteria.",
                    "2. Selected baseline delta=0.00 under parsimony rule, preserving pure router-level uncertainty awareness without second-stage additive discounting.",
                ]
            else:
                selected_candidate = min(feasible_candidates, key=lambda c: c["delta"])
                selection_rationale = [
                    f"Selected smallest feasible candidate delta={selected_candidate['delta']:.2f} under fallback parsimony."
                ]
                
        return {
            "selected_delta": selected_candidate["delta"],
            "selected_candidate_id": selected_candidate["candidate_id"],
            "selected_candidate_name": selected_candidate["name"],
            "status": "SELECTED",
            "feasible_candidate_ids": [c["candidate_id"] for c in feasible_candidates],
            "rejected_candidate_ids": [c["candidate_id"] for c in candidate_evals if c["candidate_id"] not in [f["candidate_id"] for f in feasible_candidates]],
            "audit_trail": audit_trail,
            "selection_rationale": selection_rationale,
            "selected_metrics": {
                "mean_dcri": selected_candidate["mean_dcri"],
                "median_dcri": selected_candidate["median_dcri"],
                "std_dcri": selected_candidate["std_dcri"],
                "negative_rate": selected_candidate["negative_rate"],
                "mean_penalty": selected_candidate["mean_penalty"],
                "penalty_to_base_ratio": selected_candidate["penalty_to_base_ratio"],
                "spearman_rho": selected_candidate["rank_stability"]["spearman_rho"],
                "kendall_tau": selected_candidate["rank_stability"]["kendall_tau"],
            },
        }
