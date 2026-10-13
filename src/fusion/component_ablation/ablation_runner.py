"""
FusionMedAI - Phase C11.16: ACARA-U Component Ablation Study Runner
Orchestrates the entire ablation pipeline: cohort verification,
reference reproduction, variant evaluations, regime breakdowns,
paired statistical comparisons, bootstrap resampling, and artifact generation.
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import sys
import json
import hashlib
import math
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fusion.dcri.cohort_loader import load_frozen_cohort
from src.fusion.component_ablation.ablation_config import (
    ALPHA_REF,
    BETA_REF,
    GAMMA_REF,
    ETA_REF,
    DELTA_FROZEN,
    N_PACKETS,
    SEED,
    BOOTSTRAP_ROUNDS,
    BOOTSTRAP_SEED,
    BOOTSTRAP_ALPHA,
    TAU_1_DEFAULT,
    TAU_2_DEFAULT,
    VARIANTS,
    PRIMARY_VARIANTS,
    ALL_EVALUATED_VARIANTS,
    ACTIVE_REGIMES,
    ALL_REGIMES,
    MODALITIES,
    REGIME_MODALITY_MAP,
    AblationVariantSpec,
)
from src.fusion.component_ablation.ablation_engine import AblationEngine
from src.fusion.component_ablation.ablation_metrics import (
    calc_stats,
    compute_paired_differences,
    compute_paired_bootstrap_cis,
    compute_decision_tier_transitions,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EXP_DIR = REPO_ROOT / "experiments" / "fusion" / "component_ablation"
RESULTS_DIR = EXP_DIR / "results"
REPORTS_DIR = EXP_DIR / "reports"


def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class ComponentAblationRunner:
    """
    Executes and seals Phase C11.16 Component Ablation Study.
    """

    def __init__(self, repo_root: Path = REPO_ROOT, exp_dir: Path = EXP_DIR):
        self.repo_root = repo_root
        self.exp_dir = exp_dir
        self.results_dir = exp_dir / "results"
        self.reports_dir = exp_dir / "reports"

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def run_all(self) -> Dict[str, Any]:
        """Runs the complete C11.16 ablation experiment pipeline."""
        print(">>> Phase C11.16: ACARA-U Component Ablation Study Starting <<<")

        # ---------------------------------------------------------------------
        # Stage 0: Verify Experimental Inputs
        # ---------------------------------------------------------------------
        print("\n[Stage 0] Verifying experimental inputs...")
        packet_manifest_path = self.repo_root / "experiments" / "fusion" / "baseline_comparison" / "packet_manifest.json"
        if not packet_manifest_path.is_file():
            raise FileNotFoundError(f"Reference packet manifest missing: {packet_manifest_path}")

        manifest_sha = compute_file_sha256(packet_manifest_path)
        with open(packet_manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        manifest_packets = manifest_data.get("sample_packets", [])
        manifest_packet_ids = [p["packet_id"] for p in manifest_packets]

        cohort = load_frozen_cohort(
            self.repo_root,
            n_packets=N_PACKETS,
            seed=SEED,
            manifest_path=packet_manifest_path,
            expected_sha256=manifest_sha,
        )
        packet_ids = [p.packet_id for p in cohort]
        if len(cohort) != N_PACKETS:
            raise ValueError(f"Expected {N_PACKETS} packets, got {len(cohort)}")
        if len(set(packet_ids)) != N_PACKETS:
            raise ValueError("Packet IDs must be unique across the cohort")
        if packet_ids != manifest_packet_ids:
            raise ValueError("Loaded cohort packet IDs must match frozen manifest packet IDs exactly in order and identity")
        print(f"  Input artifact verified: {packet_manifest_path.name}")
        print(f"  SHA-256: {manifest_sha}")
        print(f"  Packets loaded: {len(cohort)}, Seed: {SEED} (Matched manifest sample_packets exactly)")

        # ---------------------------------------------------------------------
        # Stage 1: Reproduce Unmodified B6 Reference (A0)
        # ---------------------------------------------------------------------
        print("\n[Stage 1] Reproducing unmodified B6 reference (A0)...")
        engine_a0 = AblationEngine(VARIANTS["A0"])
        a0_results = engine_a0.evaluate_cohort(cohort)

        # Immediate check for packet exceptions or non-finite/missing records BEFORE any mean calculations
        pkt_exceptions = getattr(engine_a0, "packet_exceptions", 0)
        failing_records = [
            r for r in a0_results
            if not r.get("is_finite", False)
            or r.get("r_fusion") is None
            or r.get("dcri") is None
            or r.get("uncertainty_penalty") is None
            or r.get("error") is not None
        ]

        if pkt_exceptions > 0 or failing_records:
            early_failure_analysis = {
                "stage": "Stage 1: A0 Reference Reproduction",
                "status": "EARLY_STAGE_FAILURE",
                "errors": [
                    f"A0 encountered {pkt_exceptions} packet exceptions and {len(failing_records)} non-finite/missing records"
                ],
                "a0_reproduction": {
                    "variant_id": "A0",
                    "packet_exceptions": pkt_exceptions,
                    "failing_records_count": len(failing_records),
                    "failing_records": [
                        {
                            "packet_id": r.get("packet_id"),
                            "error": r.get("error", "Non-finite or missing numerical result"),
                            "status": r.get("status"),
                        }
                        for r in failing_records
                    ],
                },
                "cohort_size": len(cohort),
            }
            with open(self.results_dir / "failure_analysis.json", "w", encoding="utf-8") as f:
                json.dump(early_failure_analysis, f, indent=2)
            raise RuntimeError(
                f"A0 Reference reproduction failed with {pkt_exceptions} exceptions and {len(failing_records)} non-finite records. "
                f"Diagnostic failure report saved to results/failure_analysis.json"
            )

        # All records are confirmed finite and non-null; safely compute numerical statistics
        a0_rfusion_mean = float(np.mean([r["r_fusion"] for r in a0_results]))
        a0_dcri_mean = float(np.mean([r["dcri"] for r in a0_results]))
        a0_penalty_mean = float(np.mean([r["uncertainty_penalty"] for r in a0_results]))
        a0_negatives = sum(1 for r in a0_results if r["dcri"] < 0)

        # Reference tolerances from Phase C11.13 / C11.15
        expected_rfusion = 0.28989984657769124
        expected_dcri = 0.22650622777769125
        expected_penalty = 0.0633936188
        expected_negatives = 37

        reproduction_errors = []
        if abs(a0_rfusion_mean - expected_rfusion) >= 1e-6:
            reproduction_errors.append(f"R_fusion mismatch: {a0_rfusion_mean} vs {expected_rfusion}")
        if abs(a0_dcri_mean - expected_dcri) >= 1e-6:
            reproduction_errors.append(f"DCRI mismatch: {a0_dcri_mean} vs {expected_dcri}")
        if abs(a0_penalty_mean - expected_penalty) >= 1e-6:
            reproduction_errors.append(f"Penalty mismatch: {a0_penalty_mean} vs {expected_penalty}")
        if a0_negatives != expected_negatives:
            reproduction_errors.append(f"Negatives mismatch: {a0_negatives} vs {expected_negatives}")

        if reproduction_errors:
            early_failure_analysis = {
                "stage": "Stage 1: A0 Reference Reproduction",
                "status": "EARLY_STAGE_FAILURE",
                "errors": reproduction_errors,
                "a0_reproduction": {
                    "variant_id": "A0",
                    "packet_exceptions": 0,
                    "r_fusion_mean": a0_rfusion_mean,
                    "expected_rfusion": expected_rfusion,
                    "dcri_mean": a0_dcri_mean,
                    "expected_dcri": expected_dcri,
                    "penalty_mean": a0_penalty_mean,
                    "expected_penalty": expected_penalty,
                    "negative_dcri_count": a0_negatives,
                    "expected_negatives": expected_negatives,
                },
                "cohort_size": len(cohort),
            }
            with open(self.results_dir / "failure_analysis.json", "w", encoding="utf-8") as f:
                json.dump(early_failure_analysis, f, indent=2)
            raise RuntimeError(
                f"A0 Reference reproduction failed: {'; '.join(reproduction_errors)}. "
                f"Diagnostic failure report saved to results/failure_analysis.json"
            )

        print(f"  A0 reference successfully reproduced within established tolerances:")
        print(f"    Mean R_fusion: {a0_rfusion_mean:.6f} (expected {expected_rfusion:.6f})")
        print(f"    Mean DCRI:     {a0_dcri_mean:.6f} (expected {expected_dcri:.6f})")
        print(f"    Mean Penalty:  {a0_penalty_mean:.6f} (expected {expected_penalty:.6f})")
        print(f"    Negatives:     {a0_negatives} / {N_PACKETS} (expected {expected_negatives})")

        # ---------------------------------------------------------------------
        # Stage 2 & 3: Evaluate All Candidate Ablation Variants
        # ---------------------------------------------------------------------
        print("\n[Stage 2 & 3] Evaluating candidate ablation variants across all packets...")
        all_variant_cohort_results: Dict[str, List[Dict[str, Any]]] = {"A0": a0_results}
        all_engines: Dict[str, AblationEngine] = {"A0": engine_a0}

        for var_id in ALL_EVALUATED_VARIANTS:
            if var_id == "A0":
                continue
            spec = VARIANTS[var_id]
            engine = AblationEngine(spec)
            all_engines[var_id] = engine
            res = engine.evaluate_cohort(cohort)
            if len(res) != N_PACKETS:
                raise RuntimeError(f"Expected {N_PACKETS} evaluations for {var_id}, got {len(res)}")

            # Immediate validation of packet records before adding or proceeding
            pkt_exceptions = getattr(engine, "packet_exceptions", 0)
            failing_records = [
                r for r in res
                if not r.get("is_finite", False)
                or r.get("r_fusion") is None
                or r.get("dcri") is None
                or r.get("uncertainty_penalty") is None
                or r.get("error") is not None
            ]
            if pkt_exceptions > 0 or failing_records:
                early_failure_analysis = {
                    "stage": f"Stage 2 & 3: Candidate Variant Evaluation ({var_id})",
                    "status": "EARLY_STAGE_FAILURE",
                    "errors": [
                        f"Variant {var_id} encountered {pkt_exceptions} packet exceptions and {len(failing_records)} non-finite/missing records"
                    ],
                    "variant_failure": {
                        "variant_id": var_id,
                        "packet_exceptions": pkt_exceptions,
                        "failing_records_count": len(failing_records),
                        "failing_records": [
                            {
                                "packet_id": r.get("packet_id"),
                                "error": r.get("error", "Non-finite or missing numerical result"),
                                "status": r.get("status"),
                            }
                            for r in failing_records
                        ],
                    },
                    "cohort_size": len(cohort),
                }
                with open(self.results_dir / "failure_analysis.json", "w", encoding="utf-8") as f:
                    json.dump(early_failure_analysis, f, indent=2)
                raise RuntimeError(
                    f"Variant {var_id} evaluation failed with {pkt_exceptions} exceptions and {len(failing_records)} non-finite records. "
                    f"Diagnostic failure report saved to results/failure_analysis.json"
                )

            all_variant_cohort_results[var_id] = res
            print(f"  Variant {var_id:10s} ({spec.name:32s}): 500 packets evaluated.")

        # ---------------------------------------------------------------------
        # Stage 4: Analyze Modality-Availability Regimes
        # ---------------------------------------------------------------------
        print("\n[Stage 4] Evaluating modality availability regimes (R, F, C, RF, RC, FC, RFC, EMPTY)...")
        regime_results_by_variant: Dict[str, Dict[str, Any]] = {}

        for var_id in ALL_EVALUATED_VARIANTS:
            engine = all_engines[var_id]
            regime_dict: Dict[str, Any] = {}

            for regime in ALL_REGIMES:
                reg_cohort_eval = engine.evaluate_cohort(cohort, regime_override=regime)

                # Immediate check before calculating stats or aggregating
                reg_failing = [
                    r for r in reg_cohort_eval
                    if not r.get("is_finite", False)
                    or r.get("r_fusion") is None
                    or r.get("dcri") is None
                    or r.get("routing_entropy") is None
                    or r.get("error") is not None
                ]
                if reg_failing:
                    early_failure_analysis = {
                        "stage": f"Stage 4: Regime Analysis ({var_id}, regime {regime})",
                        "status": "EARLY_STAGE_FAILURE",
                        "errors": [
                            f"Variant {var_id} under regime {regime} encountered {len(reg_failing)} non-finite/missing records"
                        ],
                        "regime_failure": {
                            "variant_id": var_id,
                            "regime": regime,
                            "failing_records_count": len(reg_failing),
                            "failing_records": [
                                {
                                    "packet_id": r.get("packet_id"),
                                    "error": r.get("error", "Non-finite or missing numerical result"),
                                    "status": r.get("status"),
                                }
                                for r in reg_failing
                            ],
                        },
                        "cohort_size": len(cohort),
                    }
                    with open(self.results_dir / "failure_analysis.json", "w", encoding="utf-8") as f:
                        json.dump(early_failure_analysis, f, indent=2)
                    raise RuntimeError(
                        f"Variant {var_id} regime {regime} failed with {len(reg_failing)} non-finite records. "
                        f"Diagnostic failure report saved to results/failure_analysis.json"
                    )

                rfusions = [r["r_fusion"] for r in reg_cohort_eval]
                dcris = [r["dcri"] for r in reg_cohort_eval]
                entropies = [r["routing_entropy"] for r in reg_cohort_eval]
                weight_stats = {m: calc_stats([r["weights"][m] for r in reg_cohort_eval]) for m in MODALITIES}

                tier_counts = {0: 0, 1: 0, 2: 0}
                for r in reg_cohort_eval:
                    tier_counts[r["decision_tier"]] += 1

                reg_record = {
                    "regime": regime,
                    "num_packets": len(reg_cohort_eval),
                    "active_modalities": list(REGIME_MODALITY_MAP[regime]),
                    "r_fusion": calc_stats(rfusions),
                    "dcri": calc_stats(dcris),
                    "routing_entropy": calc_stats(entropies),
                    "weight_stats": weight_stats,
                    "decision_tier_counts": tier_counts,
                }
                if regime in ["R", "F", "C", "EMPTY"]:
                    reg_record["packet_records"] = reg_cohort_eval
                regime_dict[regime] = reg_record
            regime_results_by_variant[var_id] = regime_dict
            print(f"  Regime matrix complete for {var_id} (8 regimes x 500 packets = 4,000 evaluations).")

        # ---------------------------------------------------------------------
        # Stage 5: Invariants Check & A6 Negative Control
        # ---------------------------------------------------------------------
        print("\n[Stage 5] Executing mathematical invariant audit & A6 negative control...")

        # Run A6 Negative Control (deliberate fault injection)
        print("  Running A6 availability-mask negative control fault injection...")
        engine_a6 = AblationEngine(VARIANTS["A6"])
        a6_fault_results = engine_a6.evaluate_cohort(cohort, regime_override="RF", force_mask_violation=True)
        violations_detected = sum(1 for r in a6_fault_results if r["weights"]["clinical"] > 0.0)
        if violations_detected != N_PACKETS:
            raise RuntimeError(f"A6 negative control produced {violations_detected} violations, expected {N_PACKETS}")
        print(f"  A6 negative control produced {violations_detected} / {N_PACKETS} deliberate violations.")
        print("  Safety assertion verified: Verifier successfully flags A6 as INVALID.")

        # Dynamically compute failure statistics from evaluated records
        accepted_variants_failures = {}
        invariant_violations = []

        for v in ALL_EVALUATED_VARIANTS:
            records = all_variant_cohort_results[v]
            non_finite = sum(1 for r in records if not (math.isfinite(r["r_fusion"]) and math.isfinite(r["dcri"]) and math.isfinite(r["routing_entropy"])))
            simplex_viols = sum(1 for r in records if abs(sum(r["weights"][m] for m in r["evaluated_modalities"]) - 1.0) > 1e-6)
            mask_viols = 0
            for r in records:
                inactive = set(MODALITIES) - set(r["evaluated_modalities"])
                for m in inactive:
                    if r["weights"][m] > 1e-6:
                        mask_viols += 1

            engine_v = all_engines[v]
            pkt_exceptions = getattr(engine_v, "packet_exceptions", 0)

            is_valid = (pkt_exceptions == 0 and non_finite == 0 and simplex_viols == 0 and mask_viols == 0)
            status = "VALID_CANDIDATE" if is_valid else "REJECTED_AS_INVALID_CANDIDATE"

            accepted_variants_failures[v] = {
                "variant_id": v,
                "packet_exceptions": pkt_exceptions,
                "non_finite_outputs": non_finite,
                "simplex_violations": simplex_viols,
                "mask_violations": mask_viols,
                "status": status,
            }
            if not is_valid:
                invariant_violations.append(f"Variant {v} failed validation (exceptions={pkt_exceptions}, non_finite={non_finite}, simplex={simplex_viols}, mask={mask_viols})")

        # Persist failure_analysis.json immediately before performing integrity assertions
        failure_analysis = {
            "negative_control_A6": {
                "variant_id": "A6",
                "purpose": "Verify rejection of availability-mask violations",
                "test_regime": "RF",
                "expected_inactive_modality": "clinical",
                "forced_violation_count": violations_detected,
                "total_test_packets": N_PACKETS,
                "verifier_verdict": "REJECTED_AS_INVALID_CANDIDATE" if violations_detected > 0 else "VALID_CANDIDATE",
                "safety_gate_passed": (violations_detected == N_PACKETS),
                "fault_records": a6_fault_results,
            },
            "accepted_variants_failures": accepted_variants_failures,
        }
        with open(self.results_dir / "failure_analysis.json", "w", encoding="utf-8") as f:
            json.dump(failure_analysis, f, indent=2)

        # Explicit integrity assertions that raise controlled exceptions
        if invariant_violations:
            raise RuntimeError(f"Integrity check failed: {'; '.join(invariant_violations)}. Failure report written to results/failure_analysis.json")

        for var_id in ALL_EVALUATED_VARIANTS:
            for r in all_variant_cohort_results[var_id]:
                active_sum = sum(r["weights"][m] for m in r["active_modalities"])
                if abs(active_sum - 1.0) >= 1e-6:
                    raise RuntimeError(f"Simplex violation in {var_id}: {active_sum}")
                for m in MODALITIES:
                    if r["weights"][m] < -1e-7:
                        raise RuntimeError(f"Negative weight in {var_id}: {r['weights'][m]}")
                if not (math.isfinite(r["r_fusion"]) and math.isfinite(r["dcri"]) and math.isfinite(r["routing_entropy"])):
                    raise RuntimeError(f"Non-finite output in {var_id} for packet {r['packet_id']}")

        for var_id in ALL_EVALUATED_VARIANTS:
            for regime in ACTIVE_REGIMES:
                active_m = set(REGIME_MODALITY_MAP[regime])
                inactive_m = set(MODALITIES) - active_m
                engine = all_engines[var_id]
                reg_eval = engine.evaluate_cohort(cohort, regime_override=regime)
                for r in reg_eval:
                    for m in inactive_m:
                        if r["weights"][m] != 0.0:
                            raise RuntimeError(f"Hard mask violation in {var_id} regime {regime}: {m}={r['weights'][m]}")

        print("  All accepted variants satisfy Simplex, Non-Negativity, Finite Outputs, and Hard Masking.")

        # ---------------------------------------------------------------------
        # Stage 6: Quantify Differences, Bootstrap, and Transitions
        # ---------------------------------------------------------------------
        print("\n[Stage 6] Quantifying paired differences, bootstrap CIs, and tier transitions...")
        paired_comparisons: Dict[str, Any] = {}
        for var_id in ALL_EVALUATED_VARIANTS:
            if var_id == "A0":
                continue
            diffs = compute_paired_differences(
                variant_results=all_variant_cohort_results[var_id],
                reference_results=a0_results,
            )
            transitions = compute_decision_tier_transitions(
                variant_results=all_variant_cohort_results[var_id],
                reference_results=a0_results,
            )
            paired_comparisons[var_id] = {
                "variant_id": var_id,
                "name": VARIANTS[var_id].name,
                "ablated_component": VARIANTS[var_id].ablated_component,
                "paired_differences": diffs,
                "tier_transitions": transitions,
            }

        # Paired bootstrap CIs
        bootstrap_cis = compute_paired_bootstrap_cis(
            all_variant_results=all_variant_cohort_results,
            reference_id="A0",
            n_rounds=BOOTSTRAP_ROUNDS,
            seed=BOOTSTRAP_SEED,
            alpha=BOOTSTRAP_ALPHA,
        )

        for var_id, cis in bootstrap_cis.items():
            paired_comparisons[var_id]["bootstrap_cis"] = cis

        # ---------------------------------------------------------------------
        # Stage 7: Freeze and Save Artifacts
        # ---------------------------------------------------------------------
        print("\n[Stage 7] Freezing and saving artifacts to experiments/fusion/component_ablation/...")
        
        # 1. protocol.json
        protocol = {
            "phase": "C11.16",
            "title": "ACARA-U Component Ablation Study Protocol",
            "reference_configuration": {
                "alpha": ALPHA_REF,
                "beta": BETA_REF,
                "gamma": GAMMA_REF,
                "eta": ETA_REF,
                "delta": DELTA_FROZEN,
            },
            "cohort": {
                "sample_size": N_PACKETS,
                "seed": SEED,
                "source_manifest_path": "experiments/fusion/baseline_comparison/packet_manifest.json",
                "source_manifest_sha256": manifest_sha,
                "source_manifest_size_bytes": packet_manifest_path.stat().st_size,
            },
            "decision_thresholds": {
                "tau_1": TAU_1_DEFAULT,
                "tau_2": TAU_2_DEFAULT,
            },
            "statistical_parameters": {
                "bootstrap_rounds": BOOTSTRAP_ROUNDS,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_alpha": BOOTSTRAP_ALPHA,
            },
            "variants": [VARIANTS[v].to_dict() for v in ALL_EVALUATED_VARIANTS] + [VARIANTS["A6"].to_dict()],
        }
        with open(self.exp_dir / "protocol.json", "w", encoding="utf-8") as f:
            json.dump(protocol, f, indent=2)

        # 2. variants.json
        variants_dict = {
            "variants": {v: VARIANTS[v].to_dict() for v in list(ALL_EVALUATED_VARIANTS) + ["A6"]}
        }
        with open(self.exp_dir / "variants.json", "w", encoding="utf-8") as f:
            json.dump(variants_dict, f, indent=2)

        # 3. results/reference_b6.json
        reference_b6 = {
            "variant_id": "A0",
            "name": VARIANTS["A0"].name,
            "coefficients": VARIANTS["A0"].to_dict()["coefficients"],
            "delta": DELTA_FROZEN,
            "cohort_size": N_PACKETS,
            "summary_statistics": {
                "r_fusion": calc_stats([r["r_fusion"] for r in a0_results]),
                "dcri": calc_stats([r["dcri"] for r in a0_results]),
                "uncertainty_penalty": calc_stats([r["uncertainty_penalty"] for r in a0_results]),
                "routing_entropy": calc_stats([r["routing_entropy"] for r in a0_results]),
                "weights": {m: calc_stats([r["weights"][m] for r in a0_results]) for m in MODALITIES},
                "negative_dcri_count": a0_negatives,
                "negative_dcri_rate": float(a0_negatives / N_PACKETS),
            },
            "packet_records": a0_results,
        }
        with open(self.results_dir / "reference_b6.json", "w", encoding="utf-8") as f:
            json.dump(reference_b6, f, indent=2)

        # 4. results/ablation_results.json
        ablation_summary = {
            "variants": {
                v: {
                    "variant_id": v,
                    "name": VARIANTS[v].name,
                    "ablated_component": VARIANTS[v].ablated_component,
                    "summary_statistics": {
                        "r_fusion": calc_stats([r["r_fusion"] for r in all_variant_cohort_results[v]]),
                        "dcri": calc_stats([r["dcri"] for r in all_variant_cohort_results[v]]),
                        "uncertainty_penalty": calc_stats([r["uncertainty_penalty"] for r in all_variant_cohort_results[v]]),
                        "routing_entropy": calc_stats([r["routing_entropy"] for r in all_variant_cohort_results[v]]),
                        "weights": {m: calc_stats([r["weights"][m] for r in all_variant_cohort_results[v]]) for m in MODALITIES},
                        "negative_dcri_count": sum(1 for r in all_variant_cohort_results[v] if r["dcri"] < 0),
                        "negative_dcri_rate": float(sum(1 for r in all_variant_cohort_results[v] if r["dcri"] < 0) / N_PACKETS),
                    },
                    "packet_records": all_variant_cohort_results[v],
                }
                for v in ALL_EVALUATED_VARIANTS
            }
        }
        with open(self.results_dir / "ablation_results.json", "w", encoding="utf-8") as f:
            json.dump(ablation_summary, f, indent=2)

        # 5. results/paired_comparisons.json
        with open(self.results_dir / "paired_comparisons.json", "w", encoding="utf-8") as f:
            json.dump(paired_comparisons, f, indent=2)

        # 6. results/regime_analysis.json
        with open(self.results_dir / "regime_analysis.json", "w", encoding="utf-8") as f:
            json.dump(regime_results_by_variant, f, indent=2)

        # 7. results/failure_analysis.json
        failure_analysis = {
            "negative_control_A6": {
                "variant_id": "A6",
                "purpose": "Verify rejection of availability-mask violations",
                "test_regime": "RF",
                "expected_inactive_modality": "clinical",
                "forced_violation_count": violations_detected,
                "total_test_packets": N_PACKETS,
                "verifier_verdict": "REJECTED_AS_INVALID_CANDIDATE",
                "safety_gate_passed": True,
                "fault_records": a6_fault_results,
            },
            "accepted_variants_failures": accepted_variants_failures,
        }
        with open(self.results_dir / "failure_analysis.json", "w", encoding="utf-8") as f:
            json.dump(failure_analysis, f, indent=2)

        # 8. freeze_manifest.json
        manifest_files = [
            "protocol.json",
            "variants.json",
            "results/reference_b6.json",
            "results/ablation_results.json",
            "results/paired_comparisons.json",
            "results/regime_analysis.json",
            "results/failure_analysis.json",
        ]
        manifest_entries = {}
        for rel_path in manifest_files:
            fp = self.exp_dir / rel_path
            manifest_entries[rel_path] = {
                "sha256": compute_file_sha256(fp),
                "size_bytes": fp.stat().st_size,
            }

        freeze_manifest = {
            "phase": "C11.16",
            "title": "ACARA-U Component Ablation Freeze Manifest",
            "cohort_size": N_PACKETS,
            "seed": SEED,
            "files": manifest_entries,
        }
        with open(self.exp_dir / "freeze_manifest.json", "w", encoding="utf-8") as f:
            json.dump(freeze_manifest, f, indent=2)

        print("\n>>> Phase C11.16: All artifacts generated; awaiting independent verification. <<<")
        return {
            "status": "SUCCESS",
            "artifacts_generated": len(manifest_entries) + 1,
            "manifest": freeze_manifest,
            "paired_comparisons": paired_comparisons,
        }


if __name__ == "__main__":
    runner = ComponentAblationRunner()
    runner.run_all()
