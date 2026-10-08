"""
verification/fusion/combination_analysis/test_tail_robustness.py
Phase C11.9: Unit tests for tier summaries, tail-vs-head contrast, and dispersion comparisons.
"""

import pytest
import numpy as np

from src.fusion.combination_analysis.tail_robustness import (
    compute_tier_summary,
    compute_tail_robustness_comparison,
    TierSummaryRecord,
    TailRobustnessComparison,
)


def test_tier_summary_aggregation():
    comb_results = {
        "RFC": {
            "status": "SUCCESS",
            "n_packets": 2,
            "r_fusion_raw": [0.2, 0.4],
            "dcri_raw": [0.1, 0.3],
            "entropy_raw": [1.0, 1.0],
            "u_sum_raw": [0.5, 0.5],
            "delta_r_raw": [0.0, 0.0],
            "delta_dcri_raw": [0.0, 0.0],
        },
        "RF": {
            "status": "SUCCESS",
            "n_packets": 2,
            "r_fusion_raw": [0.3, 0.5],
            "dcri_raw": [0.2, 0.4],
            "entropy_raw": [0.6, 0.6],
            "u_sum_raw": [0.6, 0.6],
            "delta_r_raw": [0.1, 0.1],
            "delta_dcri_raw": [0.1, 0.1],
        },
    }

    summary = compute_tier_summary("HEAD", ["RFC", "RF"], comb_results)
    assert summary.num_packets == 4
    assert np.isclose(summary.mean_r_fusion, 0.35)
    assert np.isclose(summary.mean_delta_r, 0.05)


def test_tail_robustness_comparison():
    head = TierSummaryRecord(
        tier="HEAD", num_packets=10, participating_combinations=["RFC"],
        mean_r_fusion=0.3, std_r_fusion=0.15, mean_dcri=0.2, std_dcri=0.15,
        mean_entropy=1.0, std_entropy=0.1, mean_u_sum=0.6, std_u_sum=0.2,
        mean_delta_r=0.0, std_delta_r=0.0, mean_delta_dcri=0.0, std_delta_dcri=0.0,
    )
    tail = TierSummaryRecord(
        tier="TAIL", num_packets=5, participating_combinations=["R"],
        mean_r_fusion=0.25, std_r_fusion=0.25, mean_dcri=0.25, std_dcri=0.25,
        mean_entropy=0.0, std_entropy=0.0, mean_u_sum=0.1, std_u_sum=0.1,
        mean_delta_r=0.15, std_delta_r=0.1, mean_delta_dcri=0.15, std_delta_dcri=0.1,
    )

    contrast = compute_tail_robustness_comparison("D2_TEST", head, tail)
    assert np.isclose(contrast.delta_std_r_fusion, 0.10)
    assert np.isclose(contrast.delta_mean_sensitivity_r, 0.15)
    assert contrast.delta_mean_u_sum < 0.0
