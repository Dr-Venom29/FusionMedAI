"""
FusionMedAI - Phase C11.15: Test Suite for Pipeline Contract Separation
Validates S15-13.
"""

import unittest
from src.fusion.router_sanity.pipeline_contract_checker import PipelineContractChecker


class TestPipelineContract(unittest.TestCase):
    """Unit test suite for pipeline interface separation and mathematical property decoupling."""

    @classmethod
    def setUpClass(cls):
        cls.checker = PipelineContractChecker()

    def test_s15_13_pipeline_decoupling(self):
        """S15-13: Verifies that router weight monotonicity is decoupled from fused risk directionality."""
        res = self.checker.verify_pipeline_decoupling()
        self.assertTrue(res["passed"], f"Pipeline decoupling verification failed: {res}")
        self.assertGreater(res["weight_delta"], 0.0)
        self.assertGreater(res["scenario_a_risk_delta"], 0.0)
        self.assertLess(res["scenario_b_risk_delta"], 0.0)
        self.assertGreater(res["scenario_a_dcri_delta"], 0.0)
        self.assertLess(res["scenario_b_dcri_delta"], 0.0)


if __name__ == "__main__":
    unittest.main()
