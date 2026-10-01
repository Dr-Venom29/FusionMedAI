"""Clinical Tabular Architecture Benchmarking Package (Phase C5)."""

from src.clinical.benchmarking.catboost import CatBoostModel, run_catboost_standalone
from src.clinical.benchmarking.tabnet import TabNetModel, run_tabnet_standalone
from src.clinical.benchmarking.complexity import audit_model_complexity, run_complexity_audit_standalone
from src.clinical.benchmarking.subgroup_analysis import evaluate_clinical_subgroups, run_subgroup_analysis_standalone
from src.clinical.benchmarking.hpo import run_hpo_search, run_catboost_hpo, run_tabnet_hpo, run_hpo_standalone
from src.clinical.benchmarking.benchmark import run_benchmark, build_models_suite

__all__ = [
    "CatBoostModel",
    "TabNetModel",
    "audit_model_complexity",
    "evaluate_clinical_subgroups",
    "run_hpo_search",
    "run_catboost_hpo",
    "run_tabnet_hpo",
    "run_benchmark",
    "build_models_suite",
    "run_catboost_standalone",
    "run_tabnet_standalone",
    "run_hpo_standalone",
    "run_complexity_audit_standalone",
    "run_subgroup_analysis_standalone",
]
