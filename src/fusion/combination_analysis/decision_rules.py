"""
src/fusion/combination_analysis/decision_rules.py
Phase C11.9: Pre-Declared Decision Rules for Tail Robustness Benchmarking

Defines the explicit, four-category hypothesis decision rule function for Phase C11.9.
"""

from typing import Tuple


RULE_PRACTICAL_SUPERIORITY = "PRACTICAL_SUPERIORITY_ESTABLISHED"
RULE_STATISTICALLY_SIGNIFICANT_MODEST = "STATISTICALLY_SIGNIFICANT_MODEST_EFFECT"
RULE_INCONCLUSIVE = "INCONCLUSIVE_NOT_STATISTICALLY_DISTINGUISHABLE"
RULE_INFERIOR = "INFERIOR"

ALL_DECISION_RULES = (
    RULE_PRACTICAL_SUPERIORITY,
    RULE_STATISTICALLY_SIGNIFICANT_MODEST,
    RULE_INCONCLUSIVE,
    RULE_INFERIOR,
)


def evaluate_tail_decision_rule(
    mean_delta: float,
    ci_low: float,
    ci_high: float,
    p_value: float,
    practical_threshold: float = -0.005,
    superiority_p_threshold: float = 0.001,
) -> str:
    """
    Evaluates pre-declared decision rules for Phase C11.9 tail robustness.

    Exhaustive, mutually exclusive categories:
    1. PRACTICAL_SUPERIORITY_ESTABLISHED:
       mean_delta < practical_threshold AND ci_high < 0.0 AND p_value < superiority_p_threshold
    2. STATISTICALLY_SIGNIFICANT_MODEST_EFFECT:
       mean_delta < 0.0 AND ci_high < 0.0 AND p_value < 0.05
       (and not meeting practical superiority: mean_delta >= practical_threshold OR p_value >= superiority_p_threshold)
    3. INFERIOR:
       mean_delta >= 0.0 AND ci_low > 0.0 (strictly positive difference, establishing that B6 has higher tail sensitivity than B5)
    4. INCONCLUSIVE_NOT_STATISTICALLY_DISTINGUISHABLE:
       Default unresolved / inconclusive category covering:
       - Confidence interval crossing zero (ci_low <= 0.0 <= ci_high)
       - Negative point estimate / interval but non-significant p-value (p_value >= 0.05)
       - Discordance between interval bounds and point estimate or test
    """
    # Branch 1: Practical Superiority
    if mean_delta < practical_threshold and ci_high < 0.0 and p_value < superiority_p_threshold:
        return RULE_PRACTICAL_SUPERIORITY

    # Branch 2: Statistically Significant Modest Effect (strictly negative CI, significant p < 0.05)
    if mean_delta < 0.0 and ci_high < 0.0 and p_value < 0.05:
        return RULE_STATISTICALLY_SIGNIFICANT_MODEST

    # Branch 3: Inferior (strictly positive CI excluding zero, mean delta >= 0)
    if mean_delta >= 0.0 and ci_low > 0.0:
        return RULE_INFERIOR

    # Branch 4: Inconclusive / Not Statistically Distinguishable
    # (Covers ci_low <= 0.0 <= ci_high, negative CI with p >= 0.05, and any discordant boundaries)
    return RULE_INCONCLUSIVE

