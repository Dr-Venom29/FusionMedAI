"""
Foot Ulcer Module — Explainability & Visual Attribution (Phase 10.6)
"""

from src.foot.xai.gradcam import FootGradCAM
from src.foot.xai.visualization import overlay_heatmap, render_explanation_panel
from src.foot.xai.evaluation import evaluate_pointing_game, evaluate_deletion_insertion

__all__ = [
    "FootGradCAM",
    "overlay_heatmap",
    "render_explanation_panel",
    "evaluate_pointing_game",
    "evaluate_deletion_insertion"
]
