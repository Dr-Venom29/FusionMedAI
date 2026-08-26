import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))
import src.config as config

logger = logging.getLogger("Uncertainty_RiskCoverage")

def compute_risk_coverage_curve(
    is_correct: np.ndarray,
    uncertainty_score: np.ndarray,
    higher_is_more_uncertain: bool = True
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Computes the risk-coverage curve, AURC, and E-AURC.
    
    1. Uncertainty score is oriented so that higher = more uncertain.
    2. Samples are sorted from lowest uncertainty to highest uncertainty.
    3. We reject the highest uncertainty samples first (retaining the lowest uncertainty samples).
    4. For each coverage (fraction of samples retained), risk is the error rate (1 - accuracy)
       among the retained samples.
       
    Returns:
      - coverages: np.ndarray of shape (num_samples,) from 1/M to 1.0
      - risks: np.ndarray of shape (num_samples,) containing error rates
      - aurc: Area Under the Risk-Coverage Curve
      - e_aurc: Excess AURC (AURC - AURC of optimal ranker)
    """
    num_samples = len(is_correct)
    is_incorrect = 1 - is_correct.astype(int)
    
    # Orient score so higher = more uncertain
    u = np.copy(uncertainty_score)
    if not higher_is_more_uncertain:
        if np.max(u) <= 1.0 and np.min(u) >= 0.0:
            u = 1.0 - u
        else:
            u = -u
            
    # Sort samples by uncertainty in ascending order (most certain first)
    sorted_indices = np.argsort(u)
    sorted_errors = is_incorrect[sorted_indices]
    
    # Risks and coverages
    # At index k, we retain the first k+1 samples (0-indexed)
    coverages = np.arange(1, num_samples + 1) / num_samples
    cumulative_errors = np.cumsum(sorted_errors)
    risks = cumulative_errors / np.arange(1, num_samples + 1)
    
    # Compute AURC (Area Under Risk-Coverage Curve)
    # AURC is the average risk across all coverage levels
    aurc = float(np.mean(risks))
    
    # Compute Optimal Ranker AURC
    # An optimal ranker rejects all incorrect samples first.
    # Therefore, it has 0 risk for coverages up to (num_samples - num_errors) / num_samples,
    # and then risk rises to reach the dataset-wide error rate at coverage 1.0.
    num_errors = int(np.sum(is_incorrect))
    
    if num_errors == 0:
        aurc_opt = 0.0
    else:
        # Sort optimal errors (errors go to the end, i.e., rejected first)
        opt_errors = np.zeros(num_samples)
        opt_errors[-num_errors:] = 1
        cumulative_opt_errors = np.cumsum(opt_errors)
        opt_risks = cumulative_opt_errors / np.arange(1, num_samples + 1)
        aurc_opt = float(np.mean(opt_risks))
        
    e_aurc = aurc - aurc_opt
    
    return coverages, risks, aurc, e_aurc

def evaluate_selective_prediction(
    is_correct: np.ndarray,
    uncertainty_score: np.ndarray,
    higher_is_more_uncertain: bool = True
) -> Dict[str, Any]:
    """
    Evaluates selective prediction metrics at standard coverage levels: 100%, 90%, 80%, 70%, 50%.
    """
    coverages, risks, aurc, e_aurc = compute_risk_coverage_curve(
        is_correct,
        uncertainty_score,
        higher_is_more_uncertain
    )
    
    num_samples = len(is_correct)
    
    # Find risk at specific coverage milestones
    milestones = [1.0, 0.9, 0.8, 0.7, 0.5]
    milestone_risks = {}
    
    for m in milestones:
        # Find index closest to coverage m
        idx = np.argmin(np.abs(coverages - m))
        # Actual coverage and risk
        actual_cov = coverages[idx]
        actual_risk = risks[idx]
        milestone_risks[f"risk_at_{int(m*100)}"] = float(actual_risk)
        milestone_risks[f"coverage_at_{int(m*100)}"] = float(actual_cov)
        
    return {
        "aurc": aurc,
        "e_aurc": e_aurc,
        "milestones": milestone_risks,
        "coverages": coverages.tolist(),
        "risks": risks.tolist()
    }
