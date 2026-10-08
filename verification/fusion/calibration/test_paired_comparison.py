"""
verification/fusion/calibration/test_paired_comparison.py
Tests paired comparison computations, bootstrap CI reproducibility, and hypothesis evaluation.
"""

import pytest
import numpy as np

from src.fusion.calibration.paired_calibration_analysis import (
    paired_bootstrap_ci,
    compute_paired_bootstrap_cis,
    evaluate_calibration_hypotheses,
)
from src.fusion.calibration.calibration_runner import CalibrationExperimentRunner


@pytest.fixture(scope="module")
def clean_experiment_results():
    runner = CalibrationExperimentRunner()
    cohort = runner.load_cohort()
    profiles = runner.load_modality_calibration_profiles()
    clean_res = runner.run_clean_comparison(cohort, propagate_confidence=True)
    paired_cis = compute_paired_bootstrap_cis(clean_res, n_bootstraps=500, seed=115)
    return profiles, clean_res, paired_cis


def test_paired_bootstrap_reproducibility():
    vals_a = [0.1, 0.2, 0.3, 0.4, 0.5]
    vals_b = [0.15, 0.22, 0.31, 0.45, 0.52]

    res1 = paired_bootstrap_ci(vals_a, vals_b, n_bootstraps=200, seed=115)
    res2 = paired_bootstrap_ci(vals_a, vals_b, n_bootstraps=200, seed=115)

    assert np.isclose(res1[0], res2[0], atol=1e-6)
    assert np.isclose(res1[3], res2[3], atol=1e-6)
    assert np.isclose(res1[4], res2[4], atol=1e-6)


def test_paired_differences_calculated(clean_experiment_results):
    profiles, clean_res, paired_cis = clean_experiment_results

    assert "delta_retina_weight_b5_b2" in paired_cis
    assert "delta_foot_weight_b5_b2" in paired_cis
    assert "delta_clinical_weight_b5_b2" in paired_cis
    assert "delta_rfusion_b5_b2" in paired_cis
    assert "delta_conflict_b5_b2" in paired_cis

    delta_r = paired_cis["delta_retina_weight_b5_b2"]
    assert delta_r.bootstrap_ci_lower < delta_r.bootstrap_ci_upper


def test_packet_1to1_alignment(clean_experiment_results):
    profiles, clean_res, paired_cis = clean_experiment_results
    b2_list = clean_res["packet_results"]["B2_uncalibrated_acarau"]
    b5_list = clean_res["packet_results"]["B5_calibrated_acarau"]

    assert len(b2_list) == len(b5_list) == 500
    b2_ids = [p["packet_id"] for p in b2_list]
    b5_ids = [p["packet_id"] for p in b5_list]
    assert b2_ids == b5_ids
    assert len(set(b2_ids)) == 500

