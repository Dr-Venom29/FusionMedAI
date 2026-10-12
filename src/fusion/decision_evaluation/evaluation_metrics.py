"""
FusionMedAI - Phase C11.17: Decision-Level Evaluation Metrics
Mathematical routines for statistical aggregation, conflict scoring,
paired differences, bootstrap confidence intervals, and decision tier transition matrices.
"""

from typing import Dict, Any, List, Optional, Sequence, Tuple
import math
import numpy as np

from src.fusion.decision_evaluation.evaluation_config import (
    TAU_1_DEFAULT,
    TAU_2_DEFAULT,
    TIER_0_ROUTINE,
    TIER_1_ASSESSMENT,
    TIER_2_ESCALATION,
    TIER_NAMES,
    TIER_KEYS,
)


def calc_stats(values: Sequence[Optional[float]], allow_empty: bool = False) -> Dict[str, Any]:
    """
    Computes summary statistics with defensive validation against None, NaN, and Inf.
    Raises ValueError if values sequence is empty or contains no finite values unless allow_empty is True.
    """
    valid = [
        float(v) for v in values
        if v is not None and not math.isnan(float(v)) and not math.isinf(float(v))
    ]
    if not valid:
        if not allow_empty:
            raise ValueError("calc_stats encountered empty or entirely non-finite sequence")
        return {
            "mean": None,
            "std": None,
            "median": None,
            "min": None,
            "max": None,
            "q25": None,
            "q75": None,
            "count": 0,
        }
    
    arr = np.array(valid, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
        "count": len(valid),
    }


def compute_conflict_metrics(
    active_modalities: Sequence[str],
    risks: Dict[str, float],
    weights: Dict[str, float],
    r_fusion: float,
) -> Dict[str, Any]:
    """
    Computes continuous discordance and Conflict Severity Score (CSS) across active modalities:
    - Delta_max: maximum absolute pairwise risk disagreement
    - Delta_mean: mean absolute pairwise risk disagreement
    - sigma_w: weighted risk standard deviation sqrt(sum w_i * (r_i - R_fusion)^2)
    - CSS: Conflict Severity Score (= Delta_max for multi-modality, 0.0 for singleton/empty)
    - category: LOW (<0.20), MODERATE (0.20 <= x < 0.35), HIGH (>=0.35)
    """
    num_active = len(active_modalities)
    if num_active <= 0:
        return {
            "delta_max": 0.0,
            "delta_mean": 0.0,
            "sigma_w": 0.0,
            "css": 0.0,
            "conflict_category": "NO_MODALITY_AVAILABLE",
        }
    
    if num_active == 1:
        return {
            "delta_max": 0.0,
            "delta_mean": 0.0,
            "sigma_w": 0.0,
            "css": 0.0,
            "conflict_category": "NOT_APPLICABLE",
        }
    
    # Validate that active modalities have non-null, finite values in risks and weights
    for m in active_modalities:
        if m not in risks:
            raise ValueError(f"Active modality '{m}' missing from risks dictionary")
        r_val = risks[m]
        if r_val is None or not math.isfinite(float(r_val)):
            raise ValueError(f"Active modality '{m}' has non-finite risk: {r_val}")
        if float(r_val) < -1e-6 or float(r_val) > 1.0 + 1e-6:
            raise ValueError(f"Active modality '{m}' risk out of bounds [0, 1]: {r_val}")

        if m not in weights:
            raise ValueError(f"Active modality '{m}' missing from weights dictionary")
        w_val = weights[m]
        if w_val is None or not math.isfinite(float(w_val)):
            raise ValueError(f"Active modality '{m}' has non-finite weight: {w_val}")
        if float(w_val) < -1e-6 or float(w_val) > 1.0 + 1e-6:
            raise ValueError(f"Active modality '{m}' weight out of bounds [0, 1]: {w_val}")

    if not math.isfinite(float(r_fusion)):
        raise ValueError(f"r_fusion is non-finite: {r_fusion}")

    # Compute pairwise differences
    pairwise_diffs = []
    active_list = list(active_modalities)
    for i in range(num_active):
        for j in range(i + 1, num_active):
            m_a = active_list[i]
            m_b = active_list[j]
            diff = abs(float(risks[m_a]) - float(risks[m_b]))
            pairwise_diffs.append(diff)
            
    delta_max = float(max(pairwise_diffs)) if pairwise_diffs else 0.0
    delta_mean = float(sum(pairwise_diffs) / len(pairwise_diffs)) if pairwise_diffs else 0.0
    
    # Weighted consensus variance
    var_w = sum(
        float(weights[m]) * ((float(risks[m]) - float(r_fusion)) ** 2)
        for m in active_modalities
    )
    sigma_w = float(math.sqrt(max(0.0, var_w)))
    
    # CSS equals Delta_max
    css = delta_max
    
    # Categorical classification
    if delta_max < 0.20:
        cat = "LOW"
    elif delta_max < 0.35:
        cat = "MODERATE"
    else:
        cat = "HIGH"
        
    return {
        "delta_max": delta_max,
        "delta_mean": delta_mean,
        "sigma_w": sigma_w,
        "css": css,
        "conflict_category": cat,
    }


def compute_dcri(
    r_fusion: float,
    uncertainties: Dict[str, float],
    active_modalities: Sequence[str],
    delta: float = 0.10,
) -> Tuple[float, float]:
    """
    Computes Dual-Constraint Risk Index (DCRI) using the canonical unweighted uncertainty penalty:
        P_U = delta * sum_{i in A} U_i
        DCRI = R_fusion - P_U
        
    Negative DCRI values are intentionally preserved (unclamped) per frozen research protocol.
    """
    if not active_modalities:
        return 0.0, 0.0
    
    u_sum = sum(float(uncertainties.get(m, 0.0)) for m in active_modalities)
    p_u = float(delta) * u_sum
    dcri = float(r_fusion) - p_u
    return float(dcri), float(p_u)


def assign_decision_tier(
    score: float,
    tau_1: float = TAU_1_DEFAULT,
    tau_2: float = TAU_2_DEFAULT,
) -> int:
    """
    Maps continuous score to hypothetical decision tier:
        Tier 0 (Routine Review): score < tau_1
        Tier 1 (Additional Assessment): tau_1 <= score < tau_2
        Tier 2 (Escalation): score >= tau_2
    Negative scores naturally map to Tier 0 (< tau_1).
    """
    if score < tau_1:
        return TIER_0_ROUTINE
    elif score < tau_2:
        return TIER_1_ASSESSMENT
    else:
        return TIER_2_ESCALATION


def compute_paired_differences(
    candidate_records: List[Dict[str, Any]],
    reference_records: List[Dict[str, Any]],
) -> Dict[str, List[float]]:
    """
    Computes strict paired differences: Delta = Candidate(k) - Reference(k).
    Pairs records by packet_id, enforces unique packet IDs, finite status,
    and requires all records to be successful.
    """
    if len(candidate_records) != len(reference_records):
        raise ValueError(
            f"Record length mismatch: candidate ({len(candidate_records)}) vs reference ({len(reference_records)})"
        )

    cand_map: Dict[str, Dict[str, Any]] = {}
    for r in candidate_records:
        pid = r.get("packet_id")
        if not pid:
            raise ValueError("Candidate record missing packet_id")
        if pid in cand_map:
            raise ValueError(f"Duplicate packet_id detected in candidate records: {pid}")
        if r.get("status") not in ("SUCCESS", "NO_MODALITY_AVAILABLE") or not r.get("is_finite", False):
            raise ValueError(f"Candidate record {pid} is unsuccessful or non-finite: status={r.get('status')}")
        cand_map[pid] = r

    ref_map: Dict[str, Dict[str, Any]] = {}
    for r in reference_records:
        pid = r.get("packet_id")
        if not pid:
            raise ValueError("Reference record missing packet_id")
        if pid in ref_map:
            raise ValueError(f"Duplicate packet_id detected in reference records: {pid}")
        if r.get("status") not in ("SUCCESS", "NO_MODALITY_AVAILABLE") or not r.get("is_finite", False):
            raise ValueError(f"Reference record {pid} is unsuccessful or non-finite: status={r.get('status')}")
        ref_map[pid] = r

    if set(cand_map.keys()) != set(ref_map.keys()):
        missing_in_ref = set(cand_map.keys()) - set(ref_map.keys())
        missing_in_cand = set(ref_map.keys()) - set(cand_map.keys())
        raise ValueError(
            f"Cohort packet ID set mismatch: missing in ref: {len(missing_in_ref)}, missing in cand: {len(missing_in_cand)}"
        )

    delta_rfusion: List[float] = []
    delta_dcri: List[float] = []
    delta_w_retina: List[float] = []
    delta_w_foot: List[float] = []
    delta_w_clinical: List[float] = []
    delta_entropy: List[float] = []
    delta_css: List[float] = []

    # Preserve deterministic evaluation order based on candidate records list
    for c_rec in candidate_records:
        pid = c_rec["packet_id"]
        r_rec = ref_map[pid]

        c_r = c_rec.get("r_fusion")
        r_r = r_rec.get("r_fusion")
        c_d = c_rec.get("dcri")
        r_d = r_rec.get("dcri")

        for val, name in [
            (c_r, "candidate r_fusion"),
            (r_r, "reference r_fusion"),
            (c_d, "candidate dcri"),
            (r_d, "reference dcri"),
        ]:
            if val is None or not math.isfinite(float(val)):
                raise ValueError(f"Non-finite or null {name} in packet {pid}: {val}")

        delta_rfusion.append(float(c_r) - float(r_r))
        delta_dcri.append(float(c_d) - float(r_d))

        c_w = c_rec.get("weights", {})
        r_w = r_rec.get("weights", {})
        delta_w_retina.append(float(c_w.get("retina", 0.0)) - float(r_w.get("retina", 0.0)))
        delta_w_foot.append(float(c_w.get("foot", 0.0)) - float(r_w.get("foot", 0.0)))
        delta_w_clinical.append(float(c_w.get("clinical", 0.0)) - float(r_w.get("clinical", 0.0)))

        c_ent = c_rec.get("routing_entropy", 0.0)
        r_ent = r_rec.get("routing_entropy", 0.0)
        if not math.isfinite(float(c_ent)) or not math.isfinite(float(r_ent)):
            raise ValueError(f"Non-finite routing entropy in packet {pid}")
        delta_entropy.append(float(c_ent) - float(r_ent))

        c_css = c_rec.get("css", 0.0)
        r_css = r_rec.get("css", 0.0)
        if not math.isfinite(float(c_css)) or not math.isfinite(float(r_css)):
            raise ValueError(f"Non-finite css in packet {pid}")
        delta_css.append(float(c_css) - float(r_css))

    return {
        "delta_rfusion": delta_rfusion,
        "delta_dcri": delta_dcri,
        "delta_w_retina": delta_w_retina,
        "delta_w_foot": delta_w_foot,
        "delta_w_clinical": delta_w_clinical,
        "delta_entropy": delta_entropy,
        "delta_css": delta_css,
    }


def compute_paired_bootstrap_cis(
    diffs: Sequence[float],
    n_bootstrap: int = 1000,
    seed: int = 7117,
    alpha: float = 0.05,
    allow_empty: bool = False,
) -> Dict[str, Any]:
    """
    Computes empirical percentile bootstrap confidence intervals on paired differences.
    Validates arguments, finiteness, and sample size constraints.
    """
    if n_bootstrap <= 1:
        raise ValueError(f"n_bootstrap must be > 1, got {n_bootstrap}")
    if not (0.0 < alpha < 0.5):
        raise ValueError(f"alpha must be in (0.0, 0.5), got {alpha}")

    valid: List[float] = []
    for v in diffs:
        if v is None or not math.isfinite(float(v)):
            raise ValueError(f"Non-finite difference encountered in bootstrap input: {v}")
        valid.append(float(v))

    n = len(valid)
    if n < 2:
        if not allow_empty:
            raise ValueError(f"compute_paired_bootstrap_cis requires at least 2 observations, got {n}")
        return {
            "point_estimate": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "standard_error": 0.0,
            "crosses_zero": True,
            "n_resamples": n_bootstrap,
            "seed": seed,
            "sample_size": n,
        }

    arr = np.array(valid, dtype=np.float64)
    point_est = float(np.mean(arr))

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    resample_means = np.mean(arr[indices], axis=1)

    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_lower = float(np.percentile(resample_means, lower_pct))
    ci_upper = float(np.percentile(resample_means, upper_pct))
    se = float(np.std(resample_means, ddof=1))

    crosses_zero = bool(ci_lower <= 0.0 <= ci_upper)

    return {
        "point_estimate": point_est,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "standard_error": se,
        "crosses_zero": crosses_zero,
        "n_resamples": n_bootstrap,
        "seed": seed,
        "sample_size": n,
    }


def compute_decision_tier_transitions(
    ref_tiers: Sequence[int],
    cand_tiers: Sequence[int],
) -> Dict[str, Any]:
    """
    Constructs a 3x3 transition matrix between reference and candidate decision tiers:
    Rows = Reference Tiers (0, 1, 2)
    Cols = Candidate Tiers (0, 1, 2)
    
    Calculates:
    - matrix[ref][cand]: cell counts
    - total_reclassified: count where ref != cand
    - total_escalations: count where cand > ref (upgrades)
    - total_downgrades: count where cand < ref (downgrades)
    - reclassification_rate: total_reclassified / N
    """
    if len(ref_tiers) != len(cand_tiers):
        raise ValueError(f"Length mismatch: ref ({len(ref_tiers)}) != cand ({len(cand_tiers)})")
        
    n = len(ref_tiers)
    matrix = [[0 for _ in range(3)] for _ in range(3)]
    
    total_unchanged = 0
    total_escalations = 0
    total_downgrades = 0
    
    for r, c in zip(ref_tiers, cand_tiers):
        if r not in (0, 1, 2) or c not in (0, 1, 2):
            raise ValueError(f"Invalid tier encountered: ref={r}, cand={c}")
        matrix[r][c] += 1
        if r == c:
            total_unchanged += 1
        elif c > r:
            total_escalations += 1
        else:
            total_downgrades += 1
            
    total_reclassified = total_escalations + total_downgrades
    reclass_rate = float(total_reclassified / n) if n > 0 else 0.0
    
    # Tier distribution counts
    ref_counts = [sum(matrix[r]) for r in range(3)]
    cand_counts = [sum(matrix[r][c] for r in range(3)) for c in range(3)]
    
    return {
        "matrix": matrix,
        "ref_tier_counts": {
            "tier_0_routine": ref_counts[0],
            "tier_1_assessment": ref_counts[1],
            "tier_2_escalation": ref_counts[2],
        },
        "cand_tier_counts": {
            "tier_0_routine": cand_counts[0],
            "tier_1_assessment": cand_counts[1],
            "tier_2_escalation": cand_counts[2],
        },
        "total_packets": n,
        "total_unchanged": total_unchanged,
        "total_reclassified": total_reclassified,
        "total_escalations": total_escalations,
        "total_downgrades": total_downgrades,
        "reclassification_rate": round(reclass_rate, 6),
    }


def compute_aligned_tier_transitions(
    reference_records: Sequence[Dict[str, Any]],
    candidate_records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Pairs reference and candidate records strictly by unique packet_id and
    computes the decision tier transition matrix.
    Raises ValueError if packet IDs do not align, contain duplicates, or have invalid tiers.
    """
    ref_map: Dict[str, int] = {}
    for r in reference_records:
        pid = r.get("packet_id")
        if not pid:
            raise ValueError("Reference record missing packet_id")
        if pid in ref_map:
            raise ValueError(f"Duplicate packet_id detected in reference records: {pid}")
        tier = r.get("decision_tier")
        if tier not in (0, 1, 2):
            raise ValueError(f"Invalid decision_tier {tier} in reference record {pid}")
        ref_map[pid] = tier

    cand_map: Dict[str, int] = {}
    for r in candidate_records:
        pid = r.get("packet_id")
        if not pid:
            raise ValueError("Candidate record missing packet_id")
        if pid in cand_map:
            raise ValueError(f"Duplicate packet_id detected in candidate records: {pid}")
        tier = r.get("decision_tier")
        if tier not in (0, 1, 2):
            raise ValueError(f"Invalid decision_tier {tier} in candidate record {pid}")
        cand_map[pid] = tier

    if set(ref_map.keys()) != set(cand_map.keys()):
        raise ValueError(
            f"Packet ID set mismatch: {len(set(ref_map.keys()) ^ set(cand_map.keys()))} unaligned IDs"
        )

    # Deterministic sort by packet_id
    pids = sorted(ref_map.keys())
    ref_tiers = [ref_map[pid] for pid in pids]
    cand_tiers = [cand_map[pid] for pid in pids]

    return compute_decision_tier_transitions(ref_tiers, cand_tiers)
