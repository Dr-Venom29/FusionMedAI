"""
src/fusion/degradation/degradation_engine.py
Phase C11.10: Input Degradation Orchestration Engine

Coordinates:
1. Loading the frozen N=500 ControlledDecisionPacket cohort (seed=115).
2. Experiment A: Raw input degradation and live quality evaluation across 12 operators x 4 severity levels
   via frozen quality engines (compute_retina_quality, compute_foot_quality, compute_clinical_quality).
3. Experiment B: End-to-end ACARA-U dynamic router response using live computed quality scores.
4. Quality-Authority response slope (S_QW) and Spearman alignment (rho_QW).
5. Packet-level monotonicity analysis.
6. Baseline benchmarking across B1–B6 (isolating B5 vs B6 for quality awareness).
7. Cross-modality degradation scenarios (single, pairwise, all).
8. Hard-mask invariance verification for unavailable channels.
9. 1,000-resample paired non-parametric bootstrap confidence intervals.
"""

from typing import Dict, Any, List, Tuple, Sequence, Optional
import numpy as np
from pathlib import Path
import json

from src.fusion.baselines.decision_packet import ControlledDecisionPacket, ModalityRecord, FusionResult
from src.fusion.baselines.fusion_runner import FusionRunner
from src.fusion.router.acarau_router import ACARAUv2Router
from src.fusion.router.coefficients import RouterCoefficients
from src.fusion.dcri.dcri_engine import DCRIEngine
from src.fusion.degradation.degradation_spec import (
    SEVERITY_D0_CLEAN,
    SEVERITY_D1_MILD,
    SEVERITY_D2_MODERATE,
    SEVERITY_D3_SEVERE,
    SEVERITY_LEVELS,
    MODALITY_RETINA,
    MODALITY_FOOT,
    MODALITY_CLINICAL,
    ALL_MODALITIES,
    RETINA_OPERATORS,
    FOOT_OPERATORS,
    CLINICAL_OPERATORS,
    OPERATOR_DESCRIPTIONS,
    ALL_SCENARIOS,
    SCENARIO_DEGRADED_MODALITIES,
    get_operator_params,
)
from src.fusion.degradation.image_degradation import (
    generate_benchmark_retina_image,
    generate_benchmark_foot_image,
    apply_image_degradation,
    compute_degraded_image_quality,
    apply_image_degradation_to_record,
)
from src.fusion.degradation.clinical_degradation import (
    generate_benchmark_clinical_vector,
    apply_clinical_degradation,
    compute_degraded_clinical_quality,
    apply_clinical_degradation_to_record,
)
from src.fusion.degradation.degradation_result import (
    QualityResponseRecord,
    RoutingResponseRecord,
    SlopeRecord,
    MonotonicityRecord,
    BaselineComparisonRecord,
)
from src.fusion.degradation.response_metrics import (
    compute_bootstrap_ci_1d,
    compute_paired_bootstrap_ci_diff,
    compute_quality_authority_slope,
    compute_spearman_alignment,
    compute_monotonicity_rates,
    compute_routing_entropy,
    compute_conflict_metrics,
    check_hard_mask_invariance,
)


def get_live_degraded_modality_quality(
    sample_id: str,
    modality: str,
    operator: str,
    severity: str,
    seed: int = 115,
) -> float:
    """
    Executes Experiment A path: degrades raw input and computes quality via frozen quality engine.
    """
    if severity == SEVERITY_D0_CLEAN:
        if modality == MODALITY_RETINA:
            raw_img = generate_benchmark_retina_image(sample_id, seed=seed)
            return compute_degraded_image_quality(raw_img, modality, operator, severity, seed=seed).quality
        elif modality == MODALITY_FOOT:
            raw_img = generate_benchmark_foot_image(sample_id, seed=seed)
            return compute_degraded_image_quality(raw_img, modality, operator, severity, seed=seed).quality
        elif modality == MODALITY_CLINICAL:
            raw_vec = generate_benchmark_clinical_vector(sample_id, seed=seed)
            return compute_degraded_clinical_quality(raw_vec, operator, severity, seed=seed).quality

    if modality == MODALITY_RETINA:
        raw_img = generate_benchmark_retina_image(sample_id, seed=seed)
        return compute_degraded_image_quality(raw_img, modality, operator, severity, seed=seed).quality
    elif modality == MODALITY_FOOT:
        raw_img = generate_benchmark_foot_image(sample_id, seed=seed)
        return compute_degraded_image_quality(raw_img, modality, operator, severity, seed=seed).quality
    elif modality == MODALITY_CLINICAL:
        raw_vec = generate_benchmark_clinical_vector(sample_id, seed=seed)
        return compute_degraded_clinical_quality(raw_vec, operator, severity, seed=seed).quality
    else:
        raise ValueError(f"Unknown modality: {modality}")


def apply_degradation_to_packet(
    packet: ControlledDecisionPacket,
    modality: str,
    operator: str,
    severity: str,
    live_computed_quality: Optional[float] = None,
) -> ControlledDecisionPacket:
    """Applies a specified degradation operator to a single modality channel within a packet."""
    r_rec = packet.retina
    f_rec = packet.foot
    c_rec = packet.clinical

    if modality == MODALITY_RETINA:
        r_rec = apply_image_degradation_to_record(
            r_rec, operator, severity, live_computed_quality=live_computed_quality
        )
    elif modality == MODALITY_FOOT:
        f_rec = apply_image_degradation_to_record(
            f_rec, operator, severity, live_computed_quality=live_computed_quality
        )
    elif modality == MODALITY_CLINICAL:
        c_rec = apply_clinical_degradation_to_record(
            c_rec, operator, severity, live_computed_quality=live_computed_quality
        )

    return ControlledDecisionPacket(
        packet_id=packet.packet_id,
        retina=r_rec,
        foot=f_rec,
        clinical=c_rec,
        seed=packet.seed,
        packet_type=packet.packet_type,
    )


def apply_scenario_degradation(
    packet: ControlledDecisionPacket,
    scenario: str,
    severity: str,
    default_operators: Optional[Dict[str, str]] = None,
) -> ControlledDecisionPacket:
    """Applies degradation to all modalities involved in a cross-modality scenario."""
    if default_operators is None:
        default_operators = {
            MODALITY_RETINA: RETINA_OPERATORS[0],
            MODALITY_FOOT: FOOT_OPERATORS[0],
            MODALITY_CLINICAL: CLINICAL_OPERATORS[0],
        }

    degraded_mods = SCENARIO_DEGRADED_MODALITIES.get(scenario, ())
    current_packet = packet
    for m in degraded_mods:
        op = default_operators[m]
        live_q = get_live_degraded_modality_quality(
            packet.packet_id, m, op, severity, seed=packet.seed
        )
        current_packet = apply_degradation_to_packet(
            current_packet, m, op, severity, live_computed_quality=live_q
        )
    return current_packet


class DegradationEngine:
    """
    Core orchestration engine for Phase C11.10 Input Degradation Benchmark.
    """

    def __init__(
        self,
        cohort: Sequence[ControlledDecisionPacket],
        router_coefficients: Optional[RouterCoefficients] = None,
        dcri_delta: float = 0.20,
        n_bootstrap: int = 1000,
        seed: int = 115,
    ):
        self.cohort = list(cohort)
        self.router_coefficients = (
            router_coefficients if router_coefficients is not None else RouterCoefficients.full_acarau()
        )
        self.router = ACARAUv2Router(coefficients=self.router_coefficients)
        self.dcri_engine = DCRIEngine(router=self.router)
        self.dcri_delta = dcri_delta
        self.runner = FusionRunner()
        self.n_bootstrap = n_bootstrap
        self.seed = seed

    def evaluate_quality_response(
        self,
        modality: str,
        operator: str,
    ) -> List[QualityResponseRecord]:
        """
        Experiment A: Evaluates live quality response across D0 -> D1 -> D2 -> D3
        by running actual degradation operators through frozen quality engines.
        """
        records: List[QualityResponseRecord] = []
        
        # Clean quality from live evaluation
        clean_qualities = [
            get_live_degraded_modality_quality(p.packet_id, modality, operator, SEVERITY_D0_CLEAN, seed=self.seed)
            for p in self.cohort
        ]
        mean_q_clean = float(np.mean(clean_qualities))

        for severity in SEVERITY_LEVELS:
            deg_qualities = [
                get_live_degraded_modality_quality(p.packet_id, modality, operator, severity, seed=self.seed)
                for p in self.cohort
            ]
            delta_q = [dq - cq for dq, cq in zip(deg_qualities, clean_qualities)]

            mean_deg = float(np.mean(deg_qualities))
            mean_dq = float(np.mean(delta_q))
            std_dq = float(np.std(delta_q))
            median_dq = float(np.median(delta_q))
            ci_dq = compute_bootstrap_ci_1d(delta_q, n_resamples=self.n_bootstrap, seed=self.seed)
            loss_pct = ((mean_q_clean - mean_deg) / max(mean_q_clean, 1e-6)) * 100.0

            records.append(QualityResponseRecord(
                modality=modality,
                operator=operator,
                severity=severity,
                n_packets=len(delta_q),
                mean_q_clean=mean_q_clean,
                mean_q_degraded=mean_deg,
                mean_delta_q=mean_dq,
                std_delta_q=std_dq,
                median_delta_q=median_dq,
                ci_95_delta_q=ci_dq,
                quality_loss_pct=loss_pct,
            ))
        return records

    def evaluate_routing_response(
        self,
        modality: str,
        operator: str,
    ) -> Tuple[List[RoutingResponseRecord], List[SlopeRecord], MonotonicityRecord]:
        """
        Experiment B: Evaluates ACARA-U dynamic routing authority response across D0 -> D1 -> D2 -> D3
        using live-computed degraded quality values from Experiment A.
        """
        # Baseline clean routing
        clean_evals = [self.dcri_engine.evaluate_packet(p, delta=self.dcri_delta) for p in self.cohort]
        clean_weights = [res.modality_weights[modality] for res in clean_evals]
        clean_entropies = [compute_routing_entropy(res.modality_weights) for res in clean_evals]
        mean_w_clean = float(np.mean(clean_weights))
        mean_ent_clean = float(np.mean(clean_entropies))

        routing_records: List[RoutingResponseRecord] = []
        slope_records: List[SlopeRecord] = []

        ladder_qualities: List[List[float]] = [[] for _ in range(len(self.cohort))]
        ladder_weights: List[List[float]] = [[] for _ in range(len(self.cohort))]

        for severity in SEVERITY_LEVELS:
            # Precompute live quality values for this severity level
            live_qualities = [
                get_live_degraded_modality_quality(p.packet_id, modality, operator, severity, seed=self.seed)
                for p in self.cohort
            ]

            # Construct degraded packets with live computed quality
            degraded_packets = [
                apply_degradation_to_packet(p, modality, operator, severity, live_computed_quality=lq)
                for p, lq in zip(self.cohort, live_qualities)
            ]
            deg_evals = [self.dcri_engine.evaluate_packet(p, delta=self.dcri_delta) for p in degraded_packets]
            deg_weights = [res.modality_weights[modality] for res in deg_evals]
            deg_entropies = [compute_routing_entropy(res.modality_weights) for res in deg_evals]

            # Populate ladder
            for i, (lq, res_deg) in enumerate(zip(live_qualities, deg_evals)):
                ladder_qualities[i].append(lq)
                ladder_weights[i].append(res_deg.modality_weights[modality])

            delta_w = [dw - cw for dw, cw in zip(deg_weights, clean_weights)]
            delta_ent = [de - ce for de, ce in zip(deg_entropies, clean_entropies)]
            rar = [(cw - dw) / max(cw, 1e-6) for dw, cw in zip(deg_weights, clean_weights)]

            # Total authority absorbed by other available modalities
            other_mods = [m for m in ALL_MODALITIES if m != modality]
            redistributed = [
                sum(res_deg.modality_weights[om] - res_clean.modality_weights[om] for om in other_mods)
                for res_deg, res_clean in zip(deg_evals, clean_evals)
            ]

            mean_deg_w = float(np.mean(deg_weights))
            mean_dw = float(np.mean(delta_w))
            std_dw = float(np.std(delta_w))
            median_dw = float(np.median(delta_w))
            ci_dw = compute_bootstrap_ci_1d(delta_w, n_resamples=self.n_bootstrap, seed=self.seed)

            mean_deg_ent = float(np.mean(deg_entropies))
            mean_dent = float(np.mean(delta_ent))
            mean_rar = float(np.mean(rar))
            mean_redist = float(np.mean(redistributed))

            routing_records.append(RoutingResponseRecord(
                modality=modality,
                operator=operator,
                severity=severity,
                n_packets=len(self.cohort),
                mean_w_clean=mean_w_clean,
                mean_w_degraded=mean_deg_w,
                mean_delta_w=mean_dw,
                std_delta_w=std_dw,
                median_delta_w=median_dw,
                ci_95_delta_w=ci_dw,
                mean_rar=mean_rar,
                mean_entropy_clean=mean_ent_clean,
                mean_entropy_degraded=mean_deg_ent,
                mean_delta_entropy=mean_dent,
                redistributed_authority_mean=mean_redist,
            ))

            # Quality-Authority slope
            clean_q = [
                get_live_degraded_modality_quality(p.packet_id, modality, operator, SEVERITY_D0_CLEAN, seed=self.seed)
                for p in self.cohort
            ]
            delta_q = [dq - cq for dq, cq in zip(live_qualities, clean_q)]
            mean_s, med_s, ci_s = compute_quality_authority_slope(delta_w, delta_q)

            slope_records.append(SlopeRecord(
                modality=modality,
                operator=operator,
                severity=severity,
                mean_slope=mean_s,
                median_slope=med_s,
                ci_95_slope=ci_s,
            ))

        # Monotonicity test across ladder D0->D1->D2->D3
        n_elig, q_mono, q_rate, w_mono, w_rate = compute_monotonicity_rates(
            ladder_qualities, ladder_weights
        )
        mono_record = MonotonicityRecord(
            modality=modality,
            operator=operator,
            total_eligible_packets=n_elig,
            quality_monotonic_packets=q_mono,
            quality_monotonic_rate=q_rate,
            routing_monotonic_packets=w_mono,
            routing_monotonic_rate=w_rate,
        )

        return routing_records, slope_records, mono_record

    def evaluate_baseline_comparisons(
        self,
        modality: str,
        operator: str,
    ) -> List[BaselineComparisonRecord]:
        """
        Comparative Baseline Benchmarking across B1–B6 under degradation.
        Explicitly isolates B5 vs B6 using live computed quality values.
        """
        baseline_records: List[BaselineComparisonRecord] = []
        clean_evals = {
            b_id: [self.runner.evaluate_packet(p, b_id) for p in self.cohort]
            for b_id in ("B1", "B2", "B3", "B4", "B5", "B6")
        }

        for severity in SEVERITY_LEVELS:
            live_qualities = [
                get_live_degraded_modality_quality(p.packet_id, modality, operator, severity, seed=self.seed)
                for p in self.cohort
            ]
            degraded_packets = [
                apply_degradation_to_packet(p, modality, operator, severity, live_computed_quality=lq)
                for p, lq in zip(self.cohort, live_qualities)
            ]
            deg_evals = {
                b_id: [self.runner.evaluate_packet(p, b_id) for p in degraded_packets]
                for b_id in ("B1", "B2", "B3", "B4", "B5", "B6")
            }

            # B6 vs B5 paired difference in delta_w
            b5_dw = [
                deg.weights.get(modality, 0.0) - cln.weights.get(modality, 0.0)
                for deg, cln in zip(deg_evals["B5"], clean_evals["B5"])
            ]
            b6_dw = [
                deg.weights.get(modality, 0.0) - cln.weights.get(modality, 0.0)
                for deg, cln in zip(deg_evals["B6"], clean_evals["B6"])
            ]
            mean_b6_b5_diff, ci_low_b6_b5, ci_high_b6_b5 = compute_paired_bootstrap_ci_diff(
                b6_dw, b5_dw, n_resamples=self.n_bootstrap, seed=self.seed
            )

            for b_id in ("B1", "B2", "B3", "B4", "B5", "B6"):
                b_name = self.runner.baselines[b_id].BASELINE_NAME
                dw_list = [
                    deg.weights.get(modality, 0.0) - cln.weights.get(modality, 0.0)
                    for deg, cln in zip(deg_evals[b_id], clean_evals[b_id])
                ]
                drisk_list = [
                    abs(deg.r_fusion - cln.r_fusion)
                    for deg, cln in zip(deg_evals[b_id], clean_evals[b_id])
                ]
                w_deg_list = [deg.weights.get(modality, 0.0) for deg in deg_evals[b_id]]

                baseline_records.append(BaselineComparisonRecord(
                    modality=modality,
                    operator=operator,
                    severity=severity,
                    baseline_id=b_id,
                    baseline_name=b_name,
                    mean_w_degraded=float(np.mean(w_deg_list)),
                    mean_delta_w=float(np.mean(dw_list)),
                    mean_delta_risk=float(np.mean(drisk_list)),
                    b6_vs_b5_delta_w=mean_b6_b5_diff,
                    b6_vs_b5_paired_ci=(ci_low_b6_b5, ci_high_b6_b5),
                ))

        return baseline_records

    def evaluate_cross_modality_scenarios(self) -> Dict[str, Any]:
        """Evaluates single, pairwise, and all-modality degradation scenarios."""
        results: Dict[str, Any] = {}
        clean_evals = [self.dcri_engine.evaluate_packet(p, delta=self.dcri_delta) for p in self.cohort]
        clean_rfusion = [res.r_fusion for res in clean_evals]
        clean_dcri = [res.dcri for res in clean_evals]

        for scenario in ALL_SCENARIOS:
            results[scenario] = {}
            for severity in SEVERITY_LEVELS:
                deg_packets = [
                    apply_scenario_degradation(p, scenario, severity)
                    for p in self.cohort
                ]
                deg_evals = [self.dcri_engine.evaluate_packet(p, delta=self.dcri_delta) for p in deg_packets]

                delta_r = [abs(de.r_fusion - cr) for de, cr in zip(deg_evals, clean_rfusion)]
                delta_dcri = [abs(de.dcri - cd) for de, cd in zip(deg_evals, clean_dcri)]
                entropies = [compute_routing_entropy(de.modality_weights) for de in deg_evals]

                # Conflict metrics
                conflicts = [
                    compute_conflict_metrics(de.modality_weights, {m: p.records[m].risk for m in ALL_MODALITIES}, de.active_modalities)
                    for de, p in zip(deg_evals, deg_packets)
                ]
                delta_max_list = [c[0] for c in conflicts]
                sigma_w_list = [c[2] for c in conflicts]

                results[scenario][severity] = {
                    "mean_delta_r": round(float(np.mean(delta_r)), 6),
                    "std_delta_r": round(float(np.std(delta_r)), 6),
                    "ci_95_delta_r": [round(x, 6) for x in compute_bootstrap_ci_1d(delta_r, seed=self.seed)],
                    "mean_delta_dcri": round(float(np.mean(delta_dcri)), 6),
                    "std_delta_dcri": round(float(np.std(delta_dcri)), 6),
                    "ci_95_delta_dcri": [round(x, 6) for x in compute_bootstrap_ci_1d(delta_dcri, seed=self.seed)],
                    "mean_entropy": round(float(np.mean(entropies)), 6),
                    "mean_delta_max": round(float(np.mean(delta_max_list)), 6),
                    "mean_sigma_w": round(float(np.mean(sigma_w_list)), 6),
                }

        return results
