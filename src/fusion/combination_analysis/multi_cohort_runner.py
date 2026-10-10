"""
src/fusion/combination_analysis/multi_cohort_runner.py
Phase C11.9: Multi-Cohort Tail Robustness Experiment Runner

Executes the pre-registered multi-cohort tail robustness benchmark:
- Primary Endpoint: Paired difference in tail sensitivity Delta D_tail = D_tail(B6) - D_tail(B5) under D3 (Strong Long-Tail).
  - Micro-average (primary): packet-weighted across all tail packets (N_tail=35 in D3, 75 in D2).
  - Macro-average (sensitivity): combination-weighted across tail tiers {R, F, C}.
- Secondary Endpoint: Delta D_tail under D2 (Moderate Head-Tail).
- Reference Condition: Delta D_tail under D1 (Balanced).
- Multi-Cohort Design: 30 independent seeds (401-430), N=500 packets each, total 15,000 packets per distribution.
- Statistical Inference:
  - Cohort-level bootstrap: Resamples the 30 independent synthetic cohort realizations (between-cohort variance).
  - Hierarchical cluster bootstrap: 2-stage resampling over cohorts and packets within cohorts.
  - Student's 1-sample t-test on cohort-level deltas.
- Machine-Readable Records: Full IEEE 754 64-bit floating point precision preserved without lossy rounding.
"""

import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
from scipy import stats

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fusion.dcri.cohort_loader import load_frozen_cohort
from src.fusion.combination_analysis.combination_definition import (
    NON_EMPTY_COMBINATIONS,
    COMBINATION_RFC,
    apply_combination_to_packet,
)
from src.fusion.combination_analysis.distribution_generator import (
    ALL_DISTRIBUTIONS,
    DISTRIBUTION_D1_BALANCED,
    DISTRIBUTION_D2_MODERATE_HEAD_TAIL,
    DISTRIBUTION_D3_STRONG_LONG_TAIL,
    get_distribution_config,
    generate_stratified_distribution_cohort,
    generate_controlled_synthetic_cohort,
    AssignedPacket,
)
from src.fusion.combination_analysis.head_tail_classifier import (
    classify_distribution_tiers,
)
from src.fusion.combination_analysis.baseline_evaluator import (
    evaluate_baselines_on_combinations,
    BASELINE_IDS,
)
from src.fusion.combination_analysis.decision_rules import (
    evaluate_tail_decision_rule,
    RULE_PRACTICAL_SUPERIORITY,
    RULE_STATISTICALLY_SIGNIFICANT_MODEST,
    RULE_INCONCLUSIVE,
    RULE_INFERIOR,
)


def compute_sha256(file_path: Path) -> str:
    """Computes SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class MultiCohortCombinationRunner:
    """
    Executes the multi-cohort tail-robustness benchmark.
    """

    PROTOCOL_VERSION = "1.0.0"
    COHORT_SEEDS = list(range(401, 431))  # 30 independent cohorts
    PACKETS_PER_COHORT = 500
    PRACTICAL_THRESHOLD = -0.005
    SUPERIORITY_P_THRESHOLD = 0.001
    HISTORICAL_SEED = 115

    def __init__(self, repo_root: Path = REPO_ROOT):
        self.repo_root = repo_root
        self.output_dir = repo_root / "experiments" / "fusion" / "combination_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_all(self) -> Dict[str, Any]:
        print("=" * 80)
        print("Phase C11.9: Multi-Cohort Tail Robustness Benchmark (B6 vs B5)")
        print(f"Cohort Count: S = {len(self.COHORT_SEEDS)} | N per Cohort: {self.PACKETS_PER_COHORT} | Total Packets: {len(self.COHORT_SEEDS) * self.PACKETS_PER_COHORT}")
        print("=" * 80)

        # 1. Historical Reference Evaluation (Seed 115)
        print("\n[1/5] Evaluating Historical Reference Cohort (Seed 115, N=500)...")
        hist_cohort = load_frozen_cohort(repo_root=self.repo_root, n_packets=500, seed=115)
        hist_results = self._evaluate_cohort_all_distributions(hist_cohort, seed=115, is_historical=True)
        print("  Historical Seed 115 Results:")
        for dist in (DISTRIBUTION_D2_MODERATE_HEAD_TAIL, DISTRIBUTION_D3_STRONG_LONG_TAIL):
            d_b6 = hist_results[dist]["b6_d_tail"]
            d_b5 = hist_results[dist]["b5_d_tail"]
            print(f"    {dist}: B6 D_tail = {d_b6:.6f}, B5 D_tail = {d_b5:.6f}, Diff = {d_b6 - d_b5:+.6f}")

        # 2. Multi-Cohort Confirmatory Benchmark Execution (30 Seeds)
        print(f"\n[2/5] Executing 30-Cohort Confirmatory Benchmark (Seeds {self.COHORT_SEEDS[0]}..{self.COHORT_SEEDS[-1]})...")
        cohort_evaluations: List[Dict[str, Any]] = []

        for seed in self.COHORT_SEEDS:
            cohort = generate_controlled_synthetic_cohort(seed=seed, n_packets=self.PACKETS_PER_COHORT)
            c_res = self._evaluate_cohort_all_distributions(cohort, seed=seed, is_historical=False)
            cohort_evaluations.append(c_res)
            print(f"  Processed Cohort Seed {seed}: D3 Micro Diff = {c_res[DISTRIBUTION_D3_STRONG_LONG_TAIL]['delta_d_tail']:+.6f} | Macro Diff = {c_res[DISTRIBUTION_D3_STRONG_LONG_TAIL]['delta_d_tail_macro']:+.6f}")

        # 3. Statistical Synthesis
        print("\n[3/5] Performing Statistical Synthesis (Cohort Bootstrap, Hierarchical Cluster Bootstrap, t-test)...")
        synthesis_summary = self._compute_statistical_synthesis(cohort_evaluations)

        # 4. Serialize Protocol & Results
        print("\n[4/5] Serializing structured JSON artifacts...")
        protocol_data = {
            "protocol_version": self.PROTOCOL_VERSION,
            "experiment_name": "Multi-Cohort Tail Robustness & Head-vs-Tail Evaluation",
            "phase": "C11.9",
            "primary_comparison": "B6 (Full ACARA-U) vs B5 (Uncertainty-Ablated)",
            "primary_endpoint": "Delta D_tail under D3 (Strong Long-Tail)",
            "secondary_endpoint": "Delta D_tail under D2 (Moderate Head-Tail)",
            "reference_condition": "Delta D_tail under D1 (Balanced)",
            "sample_specification": {
                "num_cohorts": len(self.COHORT_SEEDS),
                "packets_per_cohort": self.PACKETS_PER_COHORT,
                "total_packets_per_distribution": len(self.COHORT_SEEDS) * self.PACKETS_PER_COHORT,
                "cohort_seeds": self.COHORT_SEEDS,
                "historical_reference_seed": self.HISTORICAL_SEED,
            },
            "metric_definitions": {
                "d_tail_micro": "Mean absolute deviation |r_fusion(comb) - r_fusion(RFC)| across all tail packets (packet-weighted, primary).",
                "d_tail_macro": "Mean absolute deviation across tail combinations {R, F, C} (combination-weighted, sensitivity analysis).",
                "delta_d_tail": "D_tail(B6) - D_tail(B5). Negative value indicates lower tail sensitivity / greater robustness for B6.",
                "practical_superiority_threshold": self.PRACTICAL_THRESHOLD,
                "superiority_p_threshold": self.SUPERIORITY_P_THRESHOLD,
            },
            "decision_rules": {
                RULE_PRACTICAL_SUPERIORITY: f"Delta D_tail < {self.PRACTICAL_THRESHOLD} and 95% hierarchical CI strictly excludes zero and p < {self.SUPERIORITY_P_THRESHOLD}.",
                RULE_STATISTICALLY_SIGNIFICANT_MODEST: f"Delta D_tail < 0 and 95% CI strictly excludes zero and p < 0.05, but does not meet practical superiority (Delta >= {self.PRACTICAL_THRESHOLD} or p >= {self.SUPERIORITY_P_THRESHOLD}).",
                RULE_INFERIOR: "Delta D_tail >= 0 and 95% hierarchical CI strictly positive (lower bound > 0).",
                RULE_INCONCLUSIVE: "95% hierarchical CI crosses zero or inferential criteria are discordant.",
            },
        }

        with open(self.output_dir / "protocol.json", "w", encoding="utf-8") as f:
            json.dump(protocol_data, f, indent=2)

        results_data = {
            "protocol_version": self.PROTOCOL_VERSION,
            "historical_seed_115": hist_results,
            "cohort_evaluations": cohort_evaluations,
        }

        with open(self.output_dir / "multi_cohort_results.json", "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2)

        summary_data = {
            "protocol_version": self.PROTOCOL_VERSION,
            "status": "COMPLETED",
            "historical_seed_115_summary": {
                dist: {
                    "b6_d_tail": hist_results[dist]["b6_d_tail"],
                    "b5_d_tail": hist_results[dist]["b5_d_tail"],
                    "delta_d_tail": hist_results[dist]["delta_d_tail"],
                    "b6_d_tail_macro": hist_results[dist]["b6_d_tail_macro"],
                    "b5_d_tail_macro": hist_results[dist]["b5_d_tail_macro"],
                    "delta_d_tail_macro": hist_results[dist]["delta_d_tail_macro"],
                    "ci_95": hist_results[dist]["single_cohort_ci_95"],
                }
                for dist in (DISTRIBUTION_D1_BALANCED, DISTRIBUTION_D2_MODERATE_HEAD_TAIL, DISTRIBUTION_D3_STRONG_LONG_TAIL)
            },
            "multi_cohort_synthesis": synthesis_summary,
        }

        with open(self.output_dir / "multi_cohort_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        # 5. Classified Cryptographic SHA-256 Freeze Manifest
        print("\n[5/5] Generating classified cryptographic freeze manifest...")
        manifest_files = {
            "protocol.json": {
                "artifact_type": "REGENERATED_CONFIRMATORY_OUTPUT",
                "provenance": "Generated by multi_cohort_runner.py (Phase C11.9 confirmatory protocol specification)",
            },
            "multi_cohort_results.json": {
                "artifact_type": "REGENERATED_CONFIRMATORY_OUTPUT",
                "provenance": "Generated by multi_cohort_runner.py with full-precision IEEE 754 packet records across 30 cohorts",
            },
            "multi_cohort_summary.json": {
                "artifact_type": "REGENERATED_CONFIRMATORY_OUTPUT",
                "provenance": "Generated by multi_cohort_runner.py with dual cohort-level and 2-stage cluster bootstrap synthesis",
            },
            "experiment_config.json": {
                "artifact_type": "FROZEN_HISTORICAL_DEPENDENCY",
                "provenance": "Frozen Phase C11.9 baseline ladder configuration (2026-10-08)",
            },
            "baseline_comparison.json": {
                "artifact_type": "FROZEN_HISTORICAL_DEPENDENCY",
                "provenance": "Frozen Phase C11.9 single-cohort baseline evaluation record (2026-10-08)",
            },
            "distribution_configurations.json": {
                "artifact_type": "FROZEN_HISTORICAL_DEPENDENCY",
                "provenance": "Frozen Phase C11.9 combination frequency distribution definitions D1-D3 (2026-10-08)",
            },
            "tail_robustness.json": {
                "artifact_type": "FROZEN_HISTORICAL_DEPENDENCY",
                "provenance": "Frozen Phase C11.9 head-vs-tail dispersion contrast metrics (2026-10-08)",
            },
        }

        manifest = {
            "phase": "C11.9",
            "title": "Modality-Combination Distribution & Tail-Robustness Freeze Manifest",
            "cohort_count": len(self.COHORT_SEEDS),
            "total_packets": len(self.COHORT_SEEDS) * self.PACKETS_PER_COHORT,
            "files": {},
        }

        for fname, meta in manifest_files.items():
            fpath = self.output_dir / fname
            if not fpath.is_file():
                raise FileNotFoundError(f"Required artifact {fname} missing from output directory: {fpath}")
            manifest["files"][fname] = {
                "artifact_type": meta["artifact_type"],
                "sha256": compute_sha256(fpath),
                "size_bytes": fpath.stat().st_size,
                "provenance": meta["provenance"],
            }

        with open(self.output_dir / "freeze_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print("\n>>> MULTI-COHORT BENCHMARK GENERATED — VERIFICATION AND SEALING PENDING <<<")
        return summary_data

    def _evaluate_cohort_all_distributions(
        self,
        cohort: List[Any],
        seed: int,
        is_historical: bool = False,
    ) -> Dict[str, Any]:
        """Evaluates D1, D2, D3 for a single cohort, capturing full-precision packet-level records."""
        from src.fusion.baselines.fusion_runner import FusionRunner
        runner = FusionRunner()

        cohort_res: Dict[str, Any] = {"seed": seed, "is_historical": is_historical}

        # Pre-compute RFC reference risks
        rfc_refs: Dict[str, Dict[str, float]] = {}
        for pkt in cohort:
            rfc_pkt = apply_combination_to_packet(pkt, COMBINATION_RFC)
            rfc_evals = runner.evaluate_all_baselines(rfc_pkt)
            rfc_refs[pkt.packet_id] = {b_id: float(res.r_fusion) for b_id, res in rfc_evals.items()}

        for dist_id in ALL_DISTRIBUTIONS:
            cfg = get_distribution_config(dist_id)
            tiers = classify_distribution_tiers(cfg)
            assigned = generate_stratified_distribution_cohort(cohort, dist_id, seed=seed)

            # Evaluate B1-B6 standard summary
            eval_res = evaluate_baselines_on_combinations(
                assigned_cohort=assigned,
                full_cohort=cohort,
                head_combinations=tiers.head_combinations,
                tail_combinations=tiers.tail_combinations,
            )

            b6_info = eval_res["summary_table"]["B6"]
            b5_info = eval_res["summary_table"]["B5"]

            b6_d_tail_micro = float(b6_info["tail_sensitivity_d_tail"])
            b5_d_tail_micro = float(b5_info["tail_sensitivity_d_tail"])
            delta_d_tail_micro = float(b6_d_tail_micro - b5_d_tail_micro)

            # Detailed packet tracking (full float precision)
            tail_packet_records: List[Dict[str, Any]] = []
            comb_counts: Dict[str, int] = {c: 0 for c in NON_EMPTY_COMBINATIONS}
            sample_records_by_comb: Dict[str, List[Dict[str, Any]]] = {c: [] for c in NON_EMPTY_COMBINATIONS}
            simplex_violations = 0
            hard_masking_violations = 0
            max_simplex_dev = 0.0

            b6_dr_by_comb: Dict[str, List[float]] = {c: [] for c in tiers.tail_combinations}
            b5_dr_by_comb: Dict[str, List[float]] = {c: [] for c in tiers.tail_combinations}

            for ap in assigned:
                comb_id = ap.assigned_combination
                comb_counts[comb_id] += 1
                pkt = ap.packet
                orig_id = ap.original_packet_id

                b6_res = runner.evaluate_packet(pkt, "B6")
                b5_res = runner.evaluate_packet(pkt, "B5")

                # Verify simplex conservation & hard masking
                for b_name, b_out in (("B6", b6_res), ("B5", b5_res)):
                    w_sum = sum(b_out.weights.values())
                    dev = abs(w_sum - 1.0)
                    if dev > max_simplex_dev:
                        max_simplex_dev = dev
                    if dev > 1e-6:
                        simplex_violations += 1
                    for mod_name, w_val in b_out.weights.items():
                        is_active = getattr(pkt, mod_name).availability
                        if not is_active and w_val != 0.0:
                            hard_masking_violations += 1

                # Sample record collection (up to 2 per combination)
                if len(sample_records_by_comb[comb_id]) < 2:
                    sample_records_by_comb[comb_id].append({
                        "packet_id": orig_id,
                        "combination": comb_id,
                        "b6_weights": {m: float(w) for m, w in b6_res.weights.items()},
                        "b5_weights": {m: float(w) for m, w in b5_res.weights.items()},
                        "b6_r_fusion": float(b6_res.r_fusion),
                        "b5_r_fusion": float(b5_res.r_fusion),
                        "b6_rfc_ref": float(rfc_refs[orig_id]["B6"]),
                        "b5_rfc_ref": float(rfc_refs[orig_id]["B5"]),
                    })

                # Tail records (full precision)
                if comb_id in tiers.tail_combinations:
                    b6_dr = abs(rfc_refs[orig_id]["B6"] - b6_res.r_fusion)
                    b5_dr = abs(rfc_refs[orig_id]["B5"] - b5_res.r_fusion)
                    b6_dr_by_comb[comb_id].append(float(b6_dr))
                    b5_dr_by_comb[comb_id].append(float(b5_dr))

                    tail_packet_records.append({
                        "packet_id": orig_id,
                        "combination": comb_id,
                        "b6_r_fusion": float(b6_res.r_fusion),
                        "b5_r_fusion": float(b5_res.r_fusion),
                        "b6_rfc_ref": float(rfc_refs[orig_id]["B6"]),
                        "b5_rfc_ref": float(rfc_refs[orig_id]["B5"]),
                        "b6_delta_r": float(b6_dr),
                        "b5_delta_r": float(b5_dr),
                        "paired_diff": float(b6_dr - b5_dr),
                        "b6_weights": {m: float(w) for m, w in b6_res.weights.items()},
                        "b5_weights": {m: float(w) for m, w in b5_res.weights.items()},
                    })

            # Compute Macro Tail Sensitivity (Combination-weighted across tail tiers)
            if len(tiers.tail_combinations) > 0:
                b6_macro_means = [np.mean(b6_dr_by_comb[c]) for c in tiers.tail_combinations if len(b6_dr_by_comb[c]) > 0]
                b5_macro_means = [np.mean(b5_dr_by_comb[c]) for c in tiers.tail_combinations if len(b5_dr_by_comb[c]) > 0]
                b6_d_tail_macro = float(np.mean(b6_macro_means)) if b6_macro_means else 0.0
                b5_d_tail_macro = float(np.mean(b5_macro_means)) if b5_macro_means else 0.0
                delta_d_tail_macro = float(b6_d_tail_macro - b5_d_tail_macro)
            else:
                b6_d_tail_macro = 0.0
                b5_d_tail_macro = 0.0
                delta_d_tail_macro = 0.0

            # Single-cohort bootstrap CI
            paired_packet_diffs = [float(r["paired_diff"]) for r in tail_packet_records]
            single_ci = [0.0, 0.0]
            if len(paired_packet_diffs) > 1:
                rng_boot = np.random.RandomState(seed)
                arr_diff = np.asarray(paired_packet_diffs, dtype=np.float64)
                boot_means = [float(np.mean(rng_boot.choice(arr_diff, size=len(arr_diff), replace=True))) for _ in range(1000)]
                single_ci = [float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))]

            cohort_res[dist_id] = {
                "b6_d_tail": float(b6_d_tail_micro),
                "b5_d_tail": float(b5_d_tail_micro),
                "delta_d_tail": float(delta_d_tail_micro),
                "b6_d_tail_macro": float(b6_d_tail_macro),
                "b5_d_tail_macro": float(b5_d_tail_macro),
                "delta_d_tail_macro": float(delta_d_tail_macro),
                "num_tail_packets": len(tail_packet_records),
                "single_cohort_ci_95": single_ci,
                "tail_packet_diffs": paired_packet_diffs,
                "tail_packet_records": tail_packet_records,
                "combination_counts": comb_counts,
                "sample_records": sample_records_by_comb,
                "simplex_audit": {
                    "total_packets": len(assigned),
                    "simplex_violations": simplex_violations,
                    "hard_masking_violations": hard_masking_violations,
                    "max_simplex_dev": float(max_simplex_dev),
                },
                "baseline_table": {b_id: eval_res["summary_table"][b_id] for b_id in BASELINE_IDS},
            }

        return cohort_res

    # Deterministic, explicitly documented random seeds for exact reproducibility:
    # Seed scheme: BASE_SEED (11500) + DIST_OFFSET + METHOD_OFFSET
    # DIST_OFFSET: D1=0, D2=100, D3=200
    # METHOD_OFFSET: COHORT_BOOTSTRAP=1, HIERARCHICAL_BOOTSTRAP=2
    BOOTSTRAP_SEEDS = {
        (DISTRIBUTION_D1_BALANCED, "cohort"): 11501,
        (DISTRIBUTION_D1_BALANCED, "hierarchical"): 11502,
        (DISTRIBUTION_D2_MODERATE_HEAD_TAIL, "cohort"): 11601,
        (DISTRIBUTION_D2_MODERATE_HEAD_TAIL, "hierarchical"): 11602,
        (DISTRIBUTION_D3_STRONG_LONG_TAIL, "cohort"): 11701,
        (DISTRIBUTION_D3_STRONG_LONG_TAIL, "hierarchical"): 11702,
    }

    def _compute_statistical_synthesis(self, cohort_evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Computes comprehensive statistical synthesis across cohorts:
        1. Cohort-level bootstrap: Resamples the 30 independent cohort delta values.
        2. 2-stage hierarchical cluster bootstrap:
           - Micro: Resamples cohorts, then resamples tail packets within selected cohorts.
           - Macro: Resamples cohorts, then resamples tail packets within each tail combination
                    for each selected cohort, computes resampled cohort macro delta, and averages.
        3. Parametric 1-sample t-test on cohort-level deltas.
        4. Evaluates pre-declared decision rules.
        """
        synthesis: Dict[str, Any] = {}
        num_cohorts = len(cohort_evaluations)
        n_boot = 2000

        for dist_id in ALL_DISTRIBUTIONS:
            b6_d_list = [float(c[dist_id]["b6_d_tail"]) for c in cohort_evaluations]
            b5_d_list = [float(c[dist_id]["b5_d_tail"]) for c in cohort_evaluations]
            delta_d_list = [float(c[dist_id]["delta_d_tail"]) for c in cohort_evaluations]

            macro_b6_list = [float(c[dist_id]["b6_d_tail_macro"]) for c in cohort_evaluations]
            macro_b5_list = [float(c[dist_id]["b5_d_tail_macro"]) for c in cohort_evaluations]
            macro_delta_list = [float(c[dist_id]["delta_d_tail_macro"]) for c in cohort_evaluations]

            mean_b6_d = float(np.mean(b6_d_list))
            mean_b5_d = float(np.mean(b5_d_list))
            mean_delta_d = float(np.mean(delta_d_list))
            std_delta_d = float(np.std(delta_d_list, ddof=1))

            mean_b6_macro = float(np.mean(macro_b6_list))
            mean_b5_macro = float(np.mean(macro_b5_list))
            mean_delta_macro = float(np.mean(macro_delta_list))
            std_delta_macro = float(np.std(macro_delta_list, ddof=1))

            # 1. Cohort-Level Non-Parametric Bootstrap (Target A: Between-cohort variation across synthetic batches)
            cohort_seed = self.BOOTSTRAP_SEEDS[(dist_id, "cohort")]
            rng_cohort = np.random.RandomState(cohort_seed)
            arr_cohort_deltas = np.asarray(delta_d_list, dtype=np.float64)
            cohort_boot_means = np.empty(n_boot, dtype=np.float64)
            for b in range(n_boot):
                c_sample = rng_cohort.choice(arr_cohort_deltas, size=num_cohorts, replace=True)
                cohort_boot_means[b] = float(np.mean(c_sample))

            cohort_ci_low = float(np.percentile(cohort_boot_means, 2.5))
            cohort_ci_high = float(np.percentile(cohort_boot_means, 97.5))

            # 2. Two-Stage Hierarchical Cluster Bootstrap (Target B: Clustered packets nested in cohorts)
            hier_seed = self.BOOTSTRAP_SEEDS[(dist_id, "hierarchical")]
            rng_hier = np.random.RandomState(hier_seed)
            hier_boot_deltas = np.empty(n_boot, dtype=np.float64)
            hier_boot_macro = np.empty(n_boot, dtype=np.float64)

            # Pre-group tail records by combination for each cohort to optimize 2-stage macro resampling
            cfg = get_distribution_config(dist_id)
            tiers = classify_distribution_tiers(cfg)
            tail_combs = tiers.tail_combinations

            cohort_records_by_comb: List[Dict[str, List[Tuple[float, float]]]] = []
            for c in cohort_evaluations:
                c_recs = c[dist_id]["tail_packet_records"]
                comb_map: Dict[str, List[Tuple[float, float]]] = {comb: [] for comb in tail_combs}
                for r in c_recs:
                    comb_name = r["combination"]
                    if comb_name in comb_map:
                        comb_map[comb_name].append((float(r["b6_delta_r"]), float(r["b5_delta_r"])))
                cohort_records_by_comb.append(comb_map)

            for b in range(n_boot):
                cohort_indices = rng_hier.randint(0, num_cohorts, size=num_cohorts)
                resampled_cohort_micro_diffs = []
                resampled_cohort_macro_diffs = []

                for c_idx in cohort_indices:
                    # Stage 2 for Micro: resample all tail packet diffs in selected cohort
                    packet_diffs = cohort_evaluations[c_idx][dist_id]["tail_packet_diffs"]
                    if len(packet_diffs) > 0:
                        p_arr = np.asarray(packet_diffs, dtype=np.float64)
                        p_boot = rng_hier.choice(p_arr, size=len(p_arr), replace=True)
                        resampled_cohort_micro_diffs.append(float(np.mean(p_boot)))
                    else:
                        resampled_cohort_micro_diffs.append(0.0)

                    # Stage 2 for Macro: resample packets independently within each tail combination
                    comb_map = cohort_records_by_comb[c_idx]
                    comb_diff_means = []
                    for comb in tail_combs:
                        records = comb_map[comb]
                        if len(records) > 0:
                            # Resample packet pairs within this combination
                            idx_sample = rng_hier.randint(0, len(records), size=len(records))
                            b6_sample_mean = np.mean([records[i][0] for i in idx_sample])
                            b5_sample_mean = np.mean([records[i][1] for i in idx_sample])
                            comb_diff_means.append(float(b6_sample_mean - b5_sample_mean))
                    if len(comb_diff_means) > 0:
                        resampled_cohort_macro_diffs.append(float(np.mean(comb_diff_means)))
                    else:
                        resampled_cohort_macro_diffs.append(0.0)

                hier_boot_deltas[b] = float(np.mean(resampled_cohort_micro_diffs))
                hier_boot_macro[b] = float(np.mean(resampled_cohort_macro_diffs))

            hier_ci_low = float(np.percentile(hier_boot_deltas, 2.5))
            hier_ci_high = float(np.percentile(hier_boot_deltas, 97.5))

            macro_ci_low = float(np.percentile(hier_boot_macro, 2.5))
            macro_ci_high = float(np.percentile(hier_boot_macro, 97.5))

            # 3. Parametric 1-sample t-test across cohorts
            t_stat, p_val = stats.ttest_1samp(delta_d_list, 0.0)
            t_stat = float(t_stat) if not np.isnan(t_stat) else 0.0
            p_val = float(p_val) if not np.isnan(p_val) else 1.0

            # 4. Evaluate pre-declared decision rule
            if dist_id == DISTRIBUTION_D1_BALANCED:
                verdict = "REFERENCE_BALANCED_CONDITION"
            else:
                verdict = evaluate_tail_decision_rule(
                    mean_delta=mean_delta_d,
                    ci_low=hier_ci_low,
                    ci_high=hier_ci_high,
                    p_value=p_val,
                    practical_threshold=self.PRACTICAL_THRESHOLD,
                    superiority_p_threshold=self.SUPERIORITY_P_THRESHOLD,
                )

            # Generate dynamic, distribution-specific methodological explanation
            if dist_id == DISTRIBUTION_D1_BALANCED:
                justification_text = (
                    "Distribution D1 is balanced across all 7 non-empty combinations with 0 tail packets. "
                    "Delta D_tail is identically zero across all cohorts, serving as the controlled reference condition."
                )
            else:
                justification_text = (
                    f"Statistical synthesis for {dist_id} across {num_cohorts} independent cohorts: "
                    f"(1) Cohort-level estimand (Target A, between-cohort Monte Carlo variation): mean Delta = {mean_delta_d:+.6f}, "
                    f"t = {t_stat:.4f} (p = {p_val:.4f}), 95% cohort bootstrap CI: [{cohort_ci_low:+.6f}, {cohort_ci_high:+.6f}]. "
                    f"(2) 2-stage hierarchical cluster bootstrap (Target B, packets nested in cohorts): "
                    f"Micro 95% CI: [{hier_ci_low:+.6f}, {hier_ci_high:+.6f}], "
                    f"Macro 95% CI: [{macro_ci_low:+.6f}, {macro_ci_high:+.6f}]. "
                    f"Verdict evaluated under pre-declared rules: {verdict}."
                )

            synthesis[dist_id] = {
                "num_cohorts": num_cohorts,
                "mean_b6_d_tail": mean_b6_d,
                "mean_b5_d_tail": mean_b5_d,
                "mean_delta_d_tail": mean_delta_d,
                "std_delta_d_tail": std_delta_d,
                "cohort_bootstrap_ci_95": [cohort_ci_low, cohort_ci_high],
                "hierarchical_ci_95": [hier_ci_low, hier_ci_high],
                "mean_b6_d_tail_macro": mean_b6_macro,
                "mean_b5_d_tail_macro": mean_b5_macro,
                "mean_delta_d_tail_macro": mean_delta_macro,
                "std_delta_d_tail_macro": std_delta_macro,
                "hierarchical_ci_95_macro": [macro_ci_low, macro_ci_high],
                "cohort_t_statistic": round(t_stat, 4),
                "cohort_t_pvalue": float(p_val),
                "verdict": verdict,
                "methodological_justification": justification_text,
                "per_cohort_deltas": delta_d_list,
            }

        return synthesis


if __name__ == "__main__":
    runner = MultiCohortCombinationRunner()
    runner.run_all()
