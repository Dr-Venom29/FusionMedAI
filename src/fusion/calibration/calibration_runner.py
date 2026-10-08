"""
src/fusion/calibration/calibration_runner.py
Phase C11.11: Calibration Experiment Execution Engine

Orchestrates:
1. Experiment A: Clean comparison on frozen N=500 cohort (seed=115, D0) across B0–B5 conditions.
2. Experiment B: Calibration under degradation ladder (D0–D3) across 12 operators.
3. Modality-level validation calibration profiles (ECE, Brier, NLL, slopes).
4. Exporting structured JSON artifacts for analysis, verification gates, and reporting.
"""

from typing import Dict, Any, List, Tuple, Sequence, Optional
from pathlib import Path
import json
import numpy as np

from src.fusion.baselines.decision_packet import (
    ControlledDecisionPacket,
    ModalityRecord,
    FusionResult,
)
from src.fusion.baselines.fusion_runner import FusionRunner
from src.fusion.router.acarau_router import ACARAUv2Router
from src.fusion.router.coefficients import RouterCoefficients
from src.fusion.dcri.dcri_engine import DCRIEngine
from src.fusion.calibration.calibration_condition import (
    CALIBRATION_CONDITIONS,
    COND_B0_UNCAL_UNIFORM,
    COND_B1_UNCAL_RELIABILITY,
    COND_B2_UNCAL_ACARAU,
    COND_B3_CAL_UNIFORM,
    COND_B4_CAL_RELIABILITY,
    COND_B5_CAL_ACARAU,
    create_uncalibrated_decision_packet,
    create_calibrated_decision_packet,
    extract_modality_calibration_state,
)
from src.fusion.calibration.calibration_result import (
    ModalityCalibrationProfile,
    PacketCalibrationResult,
    CohortCalibrationSummary,
)
from src.fusion.calibration.calibration_metrics import (
    compute_routing_entropy,
    compute_conflict_index,
    compute_shannon_entropy,
)
from src.fusion.degradation.degradation_spec import (
    SEVERITY_D0_CLEAN,
    SEVERITY_D1_MILD,
    SEVERITY_D2_MODERATE,
    SEVERITY_D3_SEVERE,
    SEVERITY_LEVELS,
    ALL_MODALITIES,
    RETINA_OPERATORS,
    FOOT_OPERATORS,
    CLINICAL_OPERATORS,
    OP_RETINA_BLUR,
    OP_FOOT_BLUR,
    OP_CLINICAL_RANDOM_MASK,
)
from src.fusion.degradation.image_degradation import apply_image_degradation_to_record
from src.fusion.degradation.clinical_degradation import apply_clinical_degradation_to_record


class CalibrationExperimentRunner:
    """Executes C11.11 decision-level calibration benchmarks."""

    def __init__(
        self,
        cohort_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        alpha: float = 1.0,
        beta: float = 1.5,
        gamma: float = 1.0,
        eta: float = 0.5,
        delta: float = 0.20,
        seed: int = 115,
    ):
        self.root_dir = Path(__file__).resolve().parents[3]
        self.cohort_path = cohort_path or (
            self.root_dir / "experiments" / "fusion" / "dcri" / "packet_manifest.json"
        )
        self.output_dir = output_dir or (
            self.root_dir / "experiments" / "fusion" / "calibration"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.delta = delta
        self.seed = seed

        self.router_coeffs = RouterCoefficients(
            alpha=alpha, beta=beta, gamma=gamma, eta=eta
        )
        self.router = ACARAUv2Router(coefficients=self.router_coeffs)
        self.dcri_engine = DCRIEngine(router=self.router)
        self.runner = FusionRunner()

    def load_cohort(self) -> List[ControlledDecisionPacket]:
        """Loads the frozen N=500 cohort from packet manifest."""
        with open(self.cohort_path, "r") as f:
            data = json.load(f)

        packets = []
        for p in data["sample_packets"]:
            r_rec = ModalityRecord(
                sample_id=p["retina"]["sample_id"],
                modality="retina",
                risk=float(p["retina"]["risk"]),
                calibrated_probability=tuple(float(x) for x in p["retina"]["calibrated_probability"]),
                confidence=float(p["retina"]["confidence"]),
                uncertainty=float(p["retina"]["uncertainty"]),
                quality=float(p["retina"]["quality"]),
                availability=bool(p["retina"]["availability"]),
                reliability=float(p["retina"]["reliability"]),
                model_version=p["retina"]["model_version"],
            )
            f_rec = ModalityRecord(
                sample_id=p["foot"]["sample_id"],
                modality="foot",
                risk=float(p["foot"]["risk"]),
                calibrated_probability=tuple(float(x) for x in p["foot"]["calibrated_probability"]),
                confidence=float(p["foot"]["confidence"]),
                uncertainty=float(p["foot"]["uncertainty"]),
                quality=float(p["foot"]["quality"]),
                availability=bool(p["foot"]["availability"]),
                reliability=float(p["foot"]["reliability"]),
                model_version=p["foot"]["model_version"],
            )
            c_rec = ModalityRecord(
                sample_id=p["clinical"]["sample_id"],
                modality="clinical",
                risk=float(p["clinical"]["risk"]),
                calibrated_probability=tuple(float(x) for x in p["clinical"]["calibrated_probability"]),
                confidence=float(p["clinical"]["confidence"]),
                uncertainty=float(p["clinical"]["uncertainty"]),
                quality=float(p["clinical"]["quality"]),
                availability=bool(p["clinical"]["availability"]),
                reliability=float(p["clinical"]["reliability"]),
                model_version=p["clinical"]["model_version"],
            )
            packets.append(
                ControlledDecisionPacket(
                    packet_id=p["packet_id"],
                    retina=r_rec,
                    foot=f_rec,
                    clinical=c_rec,
                    seed=p["seed"],
                    packet_type="CONTROLLED_DECISION_PACKET",
                )
            )
        return packets

    def load_modality_calibration_profiles(self) -> Dict[str, ModalityCalibrationProfile]:
        """
        Loads frozen upstream single-modality validation calibration metrics from historical
        Phases C5 (Retina), C6 (Foot), and C7 (Clinical). These represent validation-anchored
        performance and are not re-derived from the synthetic C11.11 decision packets.
        """
        profiles = {}

        # 1. Retina (C5 Temperature Scaling v004 on APTOS 2019 validation split)
        profiles["retina"] = ModalityCalibrationProfile(
            modality="retina",
            calibration_method="Temperature Scaling (T=1.6218, Phase C5)",
            raw_ece=0.105843,
            calibrated_ece=0.066777,
            raw_brier=0.063140,
            calibrated_brier=0.058193,
            raw_nll=0.721966,
            calibrated_nll=0.582659,
            calibration_slope=0.9821,
            calibration_intercept=0.0012,
            raw_mean_confidence=0.902619,
            calibrated_mean_confidence=0.832434,
            raw_mean_entropy=0.2241,
            calibrated_mean_entropy=0.3487,
            ece_reduction_percent=((0.105843 - 0.066777) / 0.105843) * 100.0,
        )

        # 2. Foot (C6 Vector Scaling on ADPM V3.3 validation split)
        profiles["foot"] = ModalityCalibrationProfile(
            modality="foot",
            calibration_method="Vector Scaling (W=[1.04, 1.04, 0.87, 1.13], Phase C6)",
            raw_ece=0.087400,
            calibrated_ece=0.031300,
            raw_brier=0.074100,
            calibrated_brier=0.061200,
            raw_nll=0.942100,
            calibrated_nll=0.804400,
            calibration_slope=0.9912,
            calibration_intercept=-0.0034,
            raw_mean_confidence=0.784500,
            calibrated_mean_confidence=0.718200,
            raw_mean_entropy=0.3621,
            calibrated_mean_entropy=0.4419,
            ece_reduction_percent=((0.087400 - 0.031300) / 0.087400) * 100.0,
        )

        # 3. Clinical (C7 Platt Scaling on UCI Diabetes validation split)
        profiles["clinical"] = ModalityCalibrationProfile(
            modality="clinical",
            calibration_method="Platt / Logit Scaling (Phase C7)",
            raw_ece=0.004802,
            calibrated_ece=0.000000,
            raw_brier=0.099026,
            calibrated_brier=0.098606,
            raw_nll=0.343665,
            calibrated_nll=0.342018,
            calibration_slope=1.000183,
            calibration_intercept=0.000413,
            raw_mean_confidence=0.892400,
            calibrated_mean_confidence=0.887600,
            raw_mean_entropy=0.4120,
            calibrated_mean_entropy=0.4285,
            ece_reduction_percent=100.0,
        )

        return profiles


    def evaluate_packet_under_condition(
        self,
        packet: ControlledDecisionPacket,
        condition: str,
        propagate_confidence: bool = False,
    ) -> PacketCalibrationResult:
        """Evaluates a single packet under a specified calibration condition (B0–B5)."""
        is_calibrated = condition in (COND_B3_CAL_UNIFORM, COND_B4_CAL_RELIABILITY, COND_B5_CAL_ACARAU)
        
        # Prepare appropriate packet
        if is_calibrated:
            eval_pkt = create_calibrated_decision_packet(packet, propagate_confidence=propagate_confidence)
        else:
            eval_pkt = create_uncalibrated_decision_packet(packet, propagate_confidence=propagate_confidence)

        # Run fusion baseline
        if condition in (COND_B0_UNCAL_UNIFORM, COND_B3_CAL_UNIFORM):
            res = self.runner.evaluate_packet(eval_pkt, "B2")
        elif condition in (COND_B1_UNCAL_RELIABILITY, COND_B4_CAL_RELIABILITY):
            res = self.runner.evaluate_packet(eval_pkt, "B1")
        elif condition in (COND_B2_UNCAL_ACARAU, COND_B5_CAL_ACARAU):
            res = self.runner.evaluate_packet(eval_pkt, "B6")
        else:
            raise ValueError(f"Unknown condition: {condition}")

        weights = res.weights
        r_fusion = float(res.r_fusion)
        dcri_res = self.dcri_engine.evaluate_packet(eval_pkt, delta=self.delta)
        dcri = float(dcri_res.dcri)
        r_entropy = float(res.routing_entropy)
        risks = {m: rec.risk for m, rec in eval_pkt.records.items() if rec.availability}
        conflict = compute_conflict_index(risks)
        dom_mod = res.dominant_modality or "none"

        return PacketCalibrationResult(
            packet_id=packet.packet_id,
            condition=condition,
            weights=weights,
            r_fusion=r_fusion,
            dcri=dcri,
            routing_entropy=r_entropy,
            conflict_index=conflict,
            dominant_modality=dom_mod,
        )


    def run_clean_comparison(
        self,
        cohort: Sequence[ControlledDecisionPacket],
        propagate_confidence: bool = False,
    ) -> Dict[str, Any]:
        """
        Experiment A: Evaluates all N=500 packets across B0–B5 conditions on Clean D0 inputs.
        """
        results_by_condition: Dict[str, List[PacketCalibrationResult]] = {
            c: [] for c in CALIBRATION_CONDITIONS
        }

        for pkt in cohort:
            for cond in CALIBRATION_CONDITIONS:
                res = self.evaluate_packet_under_condition(
                    pkt, cond, propagate_confidence=propagate_confidence
                )
                results_by_condition[cond].append(res)

        # Summaries per condition
        summaries: Dict[str, CohortCalibrationSummary] = {}
        for cond, res_list in results_by_condition.items():
            is_cal = cond in (COND_B3_CAL_UNIFORM, COND_B4_CAL_RELIABILITY, COND_B5_CAL_ACARAU)
            w_r = np.mean([r.weights.get("retina", 0.0) for r in res_list])
            w_f = np.mean([r.weights.get("foot", 0.0) for r in res_list])
            w_c = np.mean([r.weights.get("clinical", 0.0) for r in res_list])
            mean_rf = float(np.mean([r.r_fusion for r in res_list]))
            std_rf = float(np.std([r.r_fusion for r in res_list]))
            mean_dcri = float(np.mean([r.dcri for r in res_list]))
            std_dcri = float(np.std([r.dcri for r in res_list]))
            mean_rent = float(np.mean([r.routing_entropy for r in res_list]))
            mean_conf = float(np.mean([r.conflict_index for r in res_list]))

            dom_counts = {"retina": 0, "foot": 0, "clinical": 0}
            for r in res_list:
                dom_counts[r.dominant_modality] += 1
            dom_dist = {k: v / len(res_list) for k, v in dom_counts.items()}

            summaries[cond] = CohortCalibrationSummary(
                condition=cond,
                is_calibrated=is_cal,
                mean_weights={"retina": float(w_r), "foot": float(w_f), "clinical": float(w_c)},
                mean_r_fusion=mean_rf,
                std_r_fusion=std_rf,
                mean_dcri=mean_dcri,
                std_dcri=std_dcri,
                mean_routing_entropy=mean_rent,
                mean_conflict_index=mean_conf,
                dominant_modality_distribution=dom_dist,
                sample_size=len(res_list),
            )

        return {
            "summaries": {k: v.to_dict() for k, v in summaries.items()},
            "packet_results": {
                k: [r.to_dict() for r in v] for k, v in results_by_condition.items()
            },
        }

    def run_degradation_comparison(
        self,
        cohort: Sequence[ControlledDecisionPacket],
        propagate_confidence: bool = False,
    ) -> Dict[str, Any]:
        """
        Experiment B: Evaluates Uncalibrated ACARA-U vs Calibrated ACARA-U across degradation ladder D0–D3.
        """
        # We test across representative degradation operators for each modality
        degradations = [
            ("retina", OP_RETINA_BLUR),
            ("foot", OP_FOOT_BLUR),
            ("clinical", OP_CLINICAL_RANDOM_MASK),
        ]

        degradation_results: Dict[str, Any] = {}

        for mod, op in degradations:
            degradation_results[op] = {}
            for sev in SEVERITY_LEVELS:
                uncal_weights = []
                cal_weights = []
                uncal_rf = []
                cal_rf = []
                uncal_dcri = []
                cal_dcri = []

                for pkt in cohort:
                    # Apply degradation to the target modality
                    if mod == "retina":
                        deg_rec = apply_image_degradation_to_record(
                            pkt.retina, op, sev
                        )
                        raw_pkt = ControlledDecisionPacket(
                            packet_id=pkt.packet_id,
                            retina=deg_rec,
                            foot=pkt.foot,
                            clinical=pkt.clinical,
                            seed=pkt.seed,
                        )
                    elif mod == "foot":
                        deg_rec = apply_image_degradation_to_record(
                            pkt.foot, op, sev
                        )
                        raw_pkt = ControlledDecisionPacket(
                            packet_id=pkt.packet_id,
                            retina=pkt.retina,
                            foot=deg_rec,
                            clinical=pkt.clinical,
                            seed=pkt.seed,
                        )
                    elif mod == "clinical":
                        deg_rec = apply_clinical_degradation_to_record(
                            pkt.clinical, op, sev
                        )
                        raw_pkt = ControlledDecisionPacket(
                            packet_id=pkt.packet_id,
                            retina=pkt.retina,
                            foot=pkt.foot,
                            clinical=deg_rec,
                            seed=pkt.seed,
                        )

                    else:
                        raise ValueError(f"Unknown modality {mod}")

                    # Evaluate Uncalibrated ACARA-U (B2)
                    res_uncal = self.evaluate_packet_under_condition(
                        raw_pkt, COND_B2_UNCAL_ACARAU, propagate_confidence=propagate_confidence
                    )
                    # Evaluate Calibrated ACARA-U (B5)
                    res_cal = self.evaluate_packet_under_condition(
                        raw_pkt, COND_B5_CAL_ACARAU, propagate_confidence=propagate_confidence
                    )

                    uncal_weights.append(res_uncal.weights[mod])
                    cal_weights.append(res_cal.weights[mod])
                    uncal_rf.append(res_uncal.r_fusion)
                    cal_rf.append(res_cal.r_fusion)
                    uncal_dcri.append(res_uncal.dcri)
                    cal_dcri.append(res_cal.dcri)

                mean_w_uncal = float(np.mean(uncal_weights))
                mean_w_cal = float(np.mean(cal_weights))
                delta_w = mean_w_cal - mean_w_uncal

                mean_rf_uncal = float(np.mean(uncal_rf))
                mean_rf_cal = float(np.mean(cal_rf))
                delta_rf = mean_rf_cal - mean_rf_uncal

                degradation_results[op][sev] = {
                    "severity": sev,
                    "target_modality": mod,
                    "mean_weight_uncalibrated": round(mean_w_uncal, 6),
                    "mean_weight_calibrated": round(mean_w_cal, 6),
                    "delta_weight": round(delta_w, 6),
                    "mean_rf_uncalibrated": round(mean_rf_uncal, 6),
                    "mean_rf_calibrated": round(mean_rf_cal, 6),
                    "delta_rf": round(delta_rf, 6),
                    "mean_dcri_uncalibrated": round(float(np.mean(uncal_dcri)), 6),
                    "mean_dcri_calibrated": round(float(np.mean(cal_dcri)), 6),
                }

        return degradation_results
