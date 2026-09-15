import warnings
from src.foot.calibration.reliability import (
    plot_reliability_diagram,
    plot_reliability_comparison,
    plot_confidence_distribution,
    plot_classwise_reliability
)

warnings.warn(
    "src.foot.calibration.visualization is deprecated; use src.foot.calibration.reliability instead.",
    DeprecationWarning,
    stacklevel=2
)
