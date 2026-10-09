"""
verification/fusion/dcri_selection/test_delta_grid.py
Tests for Phase C11.13 Candidate Delta Grid Structure, Pre-Registration, and Bounds.
"""

import unittest
import sys
from pathlib import Path

# Ensure root is in path
root_dir = Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.fusion.dcri_selection.selection_config import (
    CANDIDATE_DELTA_GRID,
    DELTA_PROVISIONAL_HISTORICAL,
)
from src.fusion.dcri_selection.candidate_grid import (
    DeltaCandidateItem,
    get_candidate_grid,
    validate_candidate_grid,
    CandidateGridError,
)


class TestDeltaCandidateGrid(unittest.TestCase):
    """Unit tests validating candidate delta grid integrity and protocol boundaries."""

    def setUp(self):
        self.grid = get_candidate_grid()

    def test_grid_length_and_endpoints(self):
        """Validates that grid contains exactly 11 candidate items spanning [0.0, 1.0]."""
        self.assertEqual(len(self.grid), 11, "Grid must contain exactly 11 candidate points")
        self.assertEqual(self.grid[0].delta, 0.0, "First point must be delta=0.0 (zero penalty)")
        self.assertEqual(self.grid[-1].delta, 1.0, "Last point must be delta=1.0 (boundary penalty)")

    def test_grid_uniqueness_and_ordering(self):
        """Validates that delta values and candidate IDs are strictly unique and monotonically increasing."""
        deltas = [item.delta for item in self.grid]
        cids = [item.candidate_id for item in self.grid]
        
        self.assertEqual(len(deltas), len(set(deltas)), "Delta values must be strictly unique")
        self.assertEqual(len(cids), len(set(cids)), "Candidate IDs must be strictly unique")
        
        for i in range(len(deltas) - 1):
            self.assertLess(deltas[i], deltas[i+1], f"Grid must be strictly monotonic: {deltas[i]} >= {deltas[i+1]}")

    def test_provisional_historical_inclusion(self):
        """Validates that historical provisional reference delta=0.20 is included."""
        prov_items = [item for item in self.grid if item.is_provisional_match]
        self.assertEqual(len(prov_items), 1, "Must contain exactly one historical provisional match")
        self.assertEqual(prov_items[0].delta, 0.20)

    def test_validation_rejects_invalid_grid(self):
        """Validates that invalid grids raise CandidateGridError."""
        bad_grid = [
            DeltaCandidateItem("D00", 0.0, "Zero", "zero_penalty", "desc", is_zero=True),
            DeltaCandidateItem("D05", 0.05, "Low", "fine_candidate", "desc"),
        ]
        with self.assertRaises(CandidateGridError):
            validate_candidate_grid(bad_grid)

    def test_validation_rejects_non_finite_and_out_of_bounds(self):
        """Validates that NaN, Inf, and out-of-bounds deltas raise CandidateGridError."""
        import math
        # Copy valid grid and corrupt one item with NaN
        nan_grid = list(self.grid)
        nan_grid[3] = DeltaCandidateItem("D15", float("nan"), "NaN", "fine_candidate", "desc")
        with self.assertRaises(CandidateGridError):
            validate_candidate_grid(nan_grid)

        # Corrupt with Inf
        inf_grid = list(self.grid)
        inf_grid[3] = DeltaCandidateItem("D15", float("inf"), "Inf", "fine_candidate", "desc")
        with self.assertRaises(CandidateGridError):
            validate_candidate_grid(inf_grid)

        # Corrupt with out-of-bounds delta (> 1.0)
        oob_grid = list(self.grid)
        oob_grid[10] = DeltaCandidateItem("D100", 1.5, "Over", "strong_penalty", "desc")
        with self.assertRaises(CandidateGridError):
            validate_candidate_grid(oob_grid)


if __name__ == "__main__":
    unittest.main()
