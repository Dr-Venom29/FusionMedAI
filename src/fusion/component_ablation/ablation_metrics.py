"""
FusionMedAI - Phase C11.16: Ablation Metrics & Statistical Analysis
Computes descriptive statistics, paired difference metrics,
paired bootstrap confidence intervals, and decision-tier transition matrices.
"""

from typing import Dict, Any, List, Tuple, Sequence, Optional
import math
import numpy as np

from src.fusion.component_ablation.ablation_config import (
    MODALITIES,
    ACTION_KEYS,
    BOOTSTRAP_ROUNDS,
    BOOTSTRAP_SEED,
    BOOTSTRAP_ALPHA,
)


def calc_stats(values: Sequence[float]) -> Dict[str, float]:
    """Calculates comprehensive descriptive statistics for a sequence of floats."""
    clean_values = [
        float(v) for v in values
        if v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
    ]
    if not clean_values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p05": 0.0,
            "q1": 0.0,
            "q3": 0.0,
            "p95": 0.0,
            "iqr": 0.0,
        }
    arr = np.array(clean_values, dtype=np.float64)
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p05": float(np.percentile(arr, 5)),
        "q1": q1,
        "q3": q3,
        "p95": float(np.percentile(arr, 95)),
        "iqr": float(q3 - q1),
    }


def compute_paired_differences(
    variant_results: List[Dict[str, Any]],
    reference_results: List[Dict[str, Any]],
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """
    Computes packet-level paired differences for a variant relative to A0 reference.
    
    Delta = Variant - A0
    """
    if len(variant_results) != len(reference_results):
        raise ValueError(f"Packet counts must match: {len(variant_results)} vs {len(reference_results)}")
    n = len(variant_results)

    delta_weights: Dict[str, List[float]] = {m: [] for m in MODALITIES}
    delta_rfusion: List[float] = []
    delta_dcri: List[float] = []
    delta_entropy: List[float] = []

    for v_rec, r_rec in zip(variant_results, reference_results):
        if v_rec["packet_id"] != r_rec["packet_id"]:
            raise ValueError(f"Packet IDs must be aligned: {v_rec['packet_id']} vs {r_rec['packet_id']}")

        for m in MODALITIES:
            delta_weights[m].append(float(v_rec["weights"][m] - r_rec["weights"][m]))

        if v_rec["r_fusion"] is None or r_rec["r_fusion"] is None:
            raise ValueError(f"Null r_fusion encountered in packet {v_rec.get('packet_id')}")
        if v_rec["dcri"] is None or r_rec["dcri"] is None:
            raise ValueError(f"Null dcri encountered in packet {v_rec.get('packet_id')}")
        if v_rec["routing_entropy"] is None or r_rec["routing_entropy"] is None:
            raise ValueError(f"Null routing_entropy encountered in packet {v_rec.get('packet_id')}")

        delta_rfusion.append(float(v_rec["r_fusion"] - r_rec["r_fusion"]))
        delta_dcri.append(float(v_rec["dcri"] - r_rec["dcri"]))
        delta_entropy.append(float(v_rec["routing_entropy"] - r_rec["routing_entropy"]))

    def classify_deltas(deltas: List[float]) -> Dict[str, Any]:
        arr = np.array(deltas, dtype=np.float64)
        pos = int(np.sum(arr > tolerance))
        neg = int(np.sum(arr < -tolerance))
        zero = int(np.sum(np.abs(arr) <= tolerance))
        return {
            "stats": calc_stats(deltas),
            "positive_count": pos,
            "negative_count": neg,
            "zero_count": zero,
            "positive_rate": float(pos / n),
            "negative_rate": float(neg / n),
            "zero_rate": float(zero / n),
        }

    return {
        "num_packets": n,
        "weight_deltas": {m: classify_deltas(delta_weights[m]) for m in MODALITIES},
        "r_fusion_delta": classify_deltas(delta_rfusion),
        "dcri_delta": classify_deltas(delta_dcri),
        "entropy_delta": classify_deltas(delta_entropy),
        "raw_deltas": {
            "packet_ids": [r["packet_id"] for r in variant_results],
            "delta_weights": delta_weights,
            "delta_rfusion": delta_rfusion,
            "delta_dcri": delta_dcri,
            "delta_entropy": delta_entropy,
        },
    }


def compute_paired_bootstrap_cis(
    all_variant_results: Dict[str, List[Dict[str, Any]]],
    reference_id: str = "A0",
    n_rounds: int = BOOTSTRAP_ROUNDS,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = BOOTSTRAP_ALPHA,
) -> Dict[str, Any]:
    """
    Computes paired bootstrap confidence intervals using identical resampled packet indices
    across all variants to preserve paired comparison integrity.
    
    Returns:
        Dictionary mapping variant_id to bootstrap CIs for mean paired differences.
    """
    ref_results = all_variant_results[reference_id]
    n_packets = len(ref_results)

    rng = np.random.RandomState(seed)
    # Generate identical resampling index matrix of shape (n_rounds, n_packets)
    resample_indices = rng.randint(0, n_packets, size=(n_rounds, n_packets))

    # Pre-extract arrays for reference
    ref_weights = {m: np.array([r["weights"][m] for r in ref_results]) for m in MODALITIES}
    ref_rfusion = np.array([r["r_fusion"] for r in ref_results])
    ref_dcri = np.array([r["dcri"] for r in ref_results])
    ref_entropy = np.array([r["routing_entropy"] for r in ref_results])

    bootstrap_results: Dict[str, Any] = {}

    for var_id, var_results in all_variant_results.items():
        if var_id == reference_id:
            continue

        var_weights = {m: np.array([r["weights"][m] for r in var_results]) for m in MODALITIES}
        var_rfusion = np.array([r["r_fusion"] for r in var_results])
        var_dcri = np.array([r["dcri"] for r in var_results])
        var_entropy = np.array([r["routing_entropy"] for r in var_results])

        # Paired diff vectors
        diff_weights = {m: var_weights[m] - ref_weights[m] for m in MODALITIES}
        diff_rfusion = var_rfusion - ref_rfusion
        diff_dcri = var_dcri - ref_dcri
        diff_entropy = var_entropy - ref_entropy

        # Bootstrap means
        boot_means_weights = {m: np.zeros(n_rounds, dtype=np.float64) for m in MODALITIES}
        boot_means_rfusion = np.zeros(n_rounds, dtype=np.float64)
        boot_means_dcri = np.zeros(n_rounds, dtype=np.float64)
        boot_means_entropy = np.zeros(n_rounds, dtype=np.float64)

        for b in range(n_rounds):
            idx = resample_indices[b]
            for m in MODALITIES:
                boot_means_weights[m][b] = np.mean(diff_weights[m][idx])
            boot_means_rfusion[b] = np.mean(diff_rfusion[idx])
            boot_means_dcri[b] = np.mean(diff_dcri[idx])
            boot_means_entropy[b] = np.mean(diff_entropy[idx])

        def get_ci(boot_arr: np.ndarray, point_est: float) -> Dict[str, float]:
            low = float(np.percentile(boot_arr, 100 * (alpha / 2.0)))
            high = float(np.percentile(boot_arr, 100 * (1.0 - alpha / 2.0)))
            se = float(np.std(boot_arr, ddof=1))
            return {
                "point_estimate": float(point_est),
                "ci_lower": low,
                "ci_upper": high,
                "std_error": se,
                "crosses_zero": bool(low <= 0.0 <= high),
            }

        var_ci: Dict[str, Any] = {
            "variant_id": var_id,
            "bootstrap_rounds": n_rounds,
            "bootstrap_seed": seed,
            "alpha": alpha,
            "r_fusion": get_ci(boot_means_rfusion, float(np.mean(diff_rfusion))),
            "dcri": get_ci(boot_means_dcri, float(np.mean(diff_dcri))),
            "routing_entropy": get_ci(boot_means_entropy, float(np.mean(diff_entropy))),
            "weights": {
                m: get_ci(boot_means_weights[m], float(np.mean(diff_weights[m])))
                for m in MODALITIES
            },
        }
        bootstrap_results[var_id] = var_ci

    return bootstrap_results


def compute_decision_tier_transitions(
    variant_results: List[Dict[str, Any]],
    reference_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Computes the 3x3 hypothetical decision tier transition matrix
    between A0 reference (rows) and the ablation variant (columns).
    
    Tiers:
        0: Routine Review
        1: Additional Assessment
        2: Escalation for Human Review
    """
    if len(variant_results) != len(reference_results):
        raise ValueError(
            f"Record count mismatch: variant={len(variant_results)}, reference={len(reference_results)}"
        )
    n = len(variant_results)

    # 3x3 matrix: matrix[ref_tier][var_tier]
    matrix = [[0 for _ in range(3)] for _ in range(3)]
    upward = 0
    downward = 0
    unchanged = 0

    ref_counts = [0, 0, 0]
    var_counts = [0, 0, 0]

    for v_rec, r_rec in zip(variant_results, reference_results):
        if v_rec["packet_id"] != r_rec["packet_id"]:
            raise ValueError(
                f"Packet ID mismatch: variant={v_rec.get('packet_id')}, reference={r_rec.get('packet_id')}"
            )
        r_tier = r_rec["decision_tier"]
        v_tier = v_rec["decision_tier"]

        ref_counts[r_tier] += 1
        var_counts[v_tier] += 1
        matrix[r_tier][v_tier] += 1

        if v_tier > r_tier:
            upward += 1
        elif v_tier < r_tier:
            downward += 1
        else:
            unchanged += 1

    return {
        "num_packets": n,
        "transition_matrix": matrix,
        "reference_tier_counts": {ACTION_KEYS[i]: ref_counts[i] for i in range(3)},
        "variant_tier_counts": {ACTION_KEYS[i]: var_counts[i] for i in range(3)},
        "unchanged_count": unchanged,
        "unchanged_rate": float(unchanged / n),
        "upward_count": upward,
        "upward_rate": float(upward / n),
        "downward_count": downward,
        "downward_rate": float(downward / n),
        "escalation_change": {
            "reference_escalation_count": ref_counts[2],
            "variant_escalation_count": var_counts[2],
            "net_escalation_change": var_counts[2] - ref_counts[2],
            "relative_escalation_change": float((var_counts[2] - ref_counts[2]) / ref_counts[2]) if ref_counts[2] > 0 else 0.0,
        },
    }
