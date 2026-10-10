"""
FusionMedAI - Outcome-Grounded Comparative Evaluation Runner
Executes paired B5 vs B6 vs B2 comparisons on outcome-grounded cohorts
and measures predictive accuracy, quality-term contribution, degradation response,
and DCRI decision loss.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import math

from src.fusion.outcome_evaluation.outcome_packet import OutcomeGroundedPacket, OutcomePacketError
from src.fusion.outcome_evaluation.outcome_metrics import (
    compute_prediction_metrics,
    assign_action_tier,
    compute_decision_loss,
)
from src.fusion.router.acarau_router import ACARAUv2Router
from src.fusion.router.coefficients import RouterCoefficients


class OutcomeEvaluationRunner:
    """
    Executes paired comparisons of B5, B6, and B2 across outcome-grounded packets.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.5,
        gamma: float = 1.0,
        eta: float = 0.5,
        delta_dcri: float = 0.10,
        tau1: float = 0.20,
        tau2: float = 0.40,
    ):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.eta = float(eta)
        self.delta_dcri = float(delta_dcri)
        self.tau1 = float(tau1)
        self.tau2 = float(tau2)

        # Routers
        # B6: Full ACARA-U with quality bonus eta
        self.router_b6 = ACARAUv2Router(
            coefficients=RouterCoefficients(alpha=self.alpha, beta=self.beta, gamma=self.gamma, eta=self.eta)
        )
        # B5: Confidence + Reliability - Uncertainty (eta=0.0)
        self.router_b5 = ACARAUv2Router(
            coefficients=RouterCoefficients(alpha=self.alpha, beta=self.beta, gamma=self.gamma, eta=0.0)
        )

    def evaluate_cohort(
        self, packets: List[OutcomeGroundedPacket]
    ) -> Dict[str, Any]:
        """
        Evaluates an entire cohort of packets and returns paired comparison results.
        """
        if len(packets) == 0:
            raise ValueError("Cannot evaluate empty packets cohort.")

        oracle_targets: List[float] = []
        oracle_tiers: List[str] = []

        b6_risks: List[float] = []
        b5_risks: List[float] = []
        b2_risks: List[float] = []

        b6_dcri_scores: List[float] = []
        b5_dcri_scores: List[float] = []

        b6_losses: List[float] = []
        b5_losses: List[float] = []
        dcri_losses: List[float] = []

        # Granular per-packet records
        packet_evaluations: List[Dict[str, Any]] = []

        # Breakdown buckets
        scenario_buckets: Dict[str, Dict[str, List[float]]] = {}
        fidelity_buckets: Dict[str, Dict[str, List[float]]] = {}
        regime_buckets: Dict[str, Dict[str, List[float]]] = {}

        # Uncertainty-stratified false downgrade buckets for high-risk packets
        unc_downgrade_buckets: Dict[str, Dict[str, int]] = {
            "LOW_UNCERTAINTY (< 0.10)": {"total": 0, "b6_fd": 0, "b5_fd": 0, "dcri_fd": 0},
            "MODERATE_UNCERTAINTY (0.10 - 0.30)": {"total": 0, "b6_fd": 0, "b5_fd": 0, "dcri_fd": 0},
            "HIGH_UNCERTAINTY (>= 0.30)": {"total": 0, "b6_fd": 0, "b5_fd": 0, "dcri_fd": 0},
        }

        for pkt in packets:
            dec_pkt = pkt.to_controlled_decision_packet()
            router_input = dec_pkt.to_router_input()
            active = dec_pkt.available_modalities

            if len(active) == 0:
                raise OutcomePacketError(f"Encountered packet {pkt.packet_id} with zero available modalities.")

            # 1. Evaluate B6
            res_b6 = self.router_b6.route(router_input)
            r_b6 = sum(res_b6.weights[m] * float(dec_pkt.records[m].risk) for m in active)

            # 2. Evaluate B5 (eta = 0.0)
            res_b5 = self.router_b5.route(router_input)
            r_b5 = sum(res_b5.weights[m] * float(dec_pkt.records[m].risk) for m in active)

            # 3. Evaluate B2 (Uniform Average)
            r_b2 = sum(float(dec_pkt.records[m].risk) for m in active) / len(active)

            # 4. DCRI scores: DCRI_delta = R_fusion - delta * sum(U_i)
            unc_list = [float(dec_pkt.records[m].uncertainty) for m in active]
            sum_unc = sum(unc_list)
            mean_unc = sum_unc / len(active)
            penalty_u = float(self.delta_dcri * sum_unc)
            dcri_b6 = float(r_b6 - penalty_u)
            dcri_b5 = float(r_b5 - penalty_u)

            # 5. Policy Tiers & Decision Losses
            tier_b6 = assign_action_tier(r_b6, self.tau1, self.tau2)
            tier_b5 = assign_action_tier(r_b5, self.tau1, self.tau2)
            tier_dcri = assign_action_tier(dcri_b6, self.tau1, self.tau2)

            loss_b6 = compute_decision_loss(tier_b6, pkt.oracle_tier)
            loss_b5 = compute_decision_loss(tier_b5, pkt.oracle_tier)
            loss_dcri = compute_decision_loss(tier_dcri, pkt.oracle_tier)

            y_true = pkt.oracle_risk
            oracle_targets.append(y_true)
            oracle_tiers.append(pkt.oracle_tier)

            b6_risks.append(r_b6)
            b5_risks.append(r_b5)
            b2_risks.append(r_b2)

            b6_dcri_scores.append(dcri_b6)
            b5_dcri_scores.append(dcri_b5)

            b6_losses.append(loss_b6)
            b5_losses.append(loss_b5)
            dcri_losses.append(loss_dcri)

            abs_err_b6 = abs(r_b6 - y_true)
            abs_err_b5 = abs(r_b5 - y_true)
            abs_err_b2 = abs(r_b2 - y_true)
            diff_mae = abs_err_b6 - abs_err_b5

            # Uncertainty stratification for High Risk cases
            if pkt.oracle_tier == "TIER_2_ESCALATION":
                if mean_unc < 0.10:
                    u_tier = "LOW_UNCERTAINTY (< 0.10)"
                elif mean_unc < 0.30:
                    u_tier = "MODERATE_UNCERTAINTY (0.10 - 0.30)"
                else:
                    u_tier = "HIGH_UNCERTAINTY (>= 0.30)"
                
                unc_downgrade_buckets[u_tier]["total"] += 1
                if tier_b6 != "TIER_2_ESCALATION":
                    unc_downgrade_buckets[u_tier]["b6_fd"] += 1
                if tier_b5 != "TIER_2_ESCALATION":
                    unc_downgrade_buckets[u_tier]["b5_fd"] += 1
                if tier_dcri != "TIER_2_ESCALATION":
                    unc_downgrade_buckets[u_tier]["dcri_fd"] += 1

            # Per-packet record
            eval_record = {
                "packet_id": pkt.packet_id,
                "seed": pkt.seed,
                "oracle_risk": round(y_true, 6),
                "oracle_tier": pkt.oracle_tier,
                "scenario": pkt.scenario,
                "quality_fidelity": pkt.quality_score_fidelity,
                "degraded_modality": pkt.degraded_modality,
                "regime": "".join(sorted([m[0].upper() for m in active])),
                "num_active": len(active),
                "mean_uncertainty": round(mean_unc, 6),
                "sum_uncertainty": round(sum_unc, 6),
                "penalty_u": round(penalty_u, 6),
                "r_b6": round(r_b6, 6),
                "r_b5": round(r_b5, 6),
                "r_b2": round(r_b2, 6),
                "dcri_b6": round(dcri_b6, 6),
                "b6_weights": {m: round(res_b6.weights[m], 6) for m in active},
                "b5_weights": {m: round(res_b5.weights[m], 6) for m in active},
                "abs_err_b6": round(abs_err_b6, 6),
                "abs_err_b5": round(abs_err_b5, 6),
                "abs_err_b2": round(abs_err_b2, 6),
                "diff_mae_b6_minus_b5": round(diff_mae, 6),
                "tier_b6": tier_b6,
                "tier_b5": tier_b5,
                "tier_dcri": tier_dcri,
                "loss_b6": loss_b6,
                "loss_b5": loss_b5,
                "loss_dcri": loss_dcri,
            }
            packet_evaluations.append(eval_record)

            # Aggregate by scenario
            scen_key = pkt.scenario
            if scen_key not in scenario_buckets:
                scenario_buckets[scen_key] = {"err_b6": [], "err_b5": [], "err_b2": []}
            scenario_buckets[scen_key]["err_b6"].append(abs_err_b6)
            scenario_buckets[scen_key]["err_b5"].append(abs_err_b5)
            scenario_buckets[scen_key]["err_b2"].append(abs_err_b2)

            # Aggregate by fidelity
            fid_key = pkt.quality_score_fidelity
            if fid_key not in fidelity_buckets:
                fidelity_buckets[fid_key] = {"err_b6": [], "err_b5": [], "err_b2": []}
            fidelity_buckets[fid_key]["err_b6"].append(abs_err_b6)
            fidelity_buckets[fid_key]["err_b5"].append(abs_err_b5)
            fidelity_buckets[fid_key]["err_b2"].append(abs_err_b2)

            # Aggregate by regime
            reg_key = eval_record["regime"]
            if reg_key not in regime_buckets:
                regime_buckets[reg_key] = {"err_b6": [], "err_b5": [], "err_b2": []}
            regime_buckets[reg_key]["err_b6"].append(abs_err_b6)
            regime_buckets[reg_key]["err_b5"].append(abs_err_b5)
            regime_buckets[reg_key]["err_b2"].append(abs_err_b2)

        # Overall Metrics
        metrics_b6 = compute_prediction_metrics(b6_risks, oracle_targets)
        metrics_b5 = compute_prediction_metrics(b5_risks, oracle_targets)
        metrics_b2 = compute_prediction_metrics(b2_risks, oracle_targets)

        # Paired differences per packet
        diffs_mae = [rec["diff_mae_b6_minus_b5"] for rec in packet_evaluations]
        mean_diff = float(np.mean(diffs_mae))

        # Scenario summaries
        scenario_summary: Dict[str, Any] = {}
        for scen, bdata in scenario_buckets.items():
            scen_mae_b6 = float(np.mean(bdata["err_b6"]))
            scen_mae_b5 = float(np.mean(bdata["err_b5"]))
            scenario_summary[scen] = {
                "count": len(bdata["err_b6"]),
                "mae_b6": round(scen_mae_b6, 6),
                "mae_b5": round(scen_mae_b5, 6),
                "delta_mae": round(scen_mae_b6 - scen_mae_b5, 6),
            }

        # Fidelity summaries
        fidelity_summary: Dict[str, Any] = {}
        for fid, bdata in fidelity_buckets.items():
            fid_mae_b6 = float(np.mean(bdata["err_b6"]))
            fid_mae_b5 = float(np.mean(bdata["err_b5"]))
            fidelity_summary[fid] = {
                "count": len(bdata["err_b6"]),
                "mae_b6": round(fid_mae_b6, 6),
                "mae_b5": round(fid_mae_b5, 6),
                "delta_mae": round(fid_mae_b6 - fid_mae_b5, 6),
            }

        # Regime summaries
        regime_summary: Dict[str, Any] = {}
        for reg, bdata in regime_buckets.items():
            reg_mae_b6 = float(np.mean(bdata["err_b6"]))
            reg_mae_b5 = float(np.mean(bdata["err_b5"]))
            regime_summary[reg] = {
                "count": len(bdata["err_b6"]),
                "mae_b6": round(reg_mae_b6, 6),
                "mae_b5": round(reg_mae_b5, 6),
                "delta_mae": round(reg_mae_b6 - reg_mae_b5, 6),
            }

        # Policy & Decision Loss Metrics
        mean_loss_b6 = float(np.mean(b6_losses))
        mean_loss_b5 = float(np.mean(b5_losses))
        mean_loss_dcri = float(np.mean(dcri_losses))

        # False Downgrades of Oracle High-Risk (TIER_2_ESCALATION)
        high_risk_indices = [i for i, t in enumerate(oracle_tiers) if t == "TIER_2_ESCALATION"]
        n_high_risk = len(high_risk_indices)

        false_downgrades_b6 = sum(1 for i in high_risk_indices if assign_action_tier(b6_risks[i], self.tau1, self.tau2) != "TIER_2_ESCALATION")
        false_downgrades_b5 = sum(1 for i in high_risk_indices if assign_action_tier(b5_risks[i], self.tau1, self.tau2) != "TIER_2_ESCALATION")
        false_downgrades_dcri = sum(1 for i in high_risk_indices if assign_action_tier(b6_dcri_scores[i], self.tau1, self.tau2) != "TIER_2_ESCALATION")

        return {
            "total_packets": len(packet_evaluations),
            "b6_metrics": metrics_b6,
            "b5_metrics": metrics_b5,
            "b2_metrics": metrics_b2,
            "mean_delta_mae": round(mean_diff, 6),
            "packet_diffs": diffs_mae,
            "scenario_breakdown": scenario_summary,
            "fidelity_breakdown": fidelity_summary,
            "regime_breakdown": regime_summary,
            "uncertainty_downgrades": unc_downgrade_buckets,
            "decision_policy": {
                "mean_loss_b6": round(mean_loss_b6, 6),
                "mean_loss_b5": round(mean_loss_b5, 6),
                "mean_loss_dcri": round(mean_loss_dcri, 6),
                "delta_loss_b6_vs_b5": round(mean_loss_b6 - mean_loss_b5, 6),
                "delta_loss_dcri_vs_b6": round(mean_loss_dcri - mean_loss_b6, 6),
                "high_risk_count": n_high_risk,
                "false_downgrades_b6": false_downgrades_b6,
                "false_downgrades_b5": false_downgrades_b5,
                "false_downgrades_dcri": false_downgrades_dcri,
                "false_downgrade_rate_b6": round(false_downgrades_b6 / max(n_high_risk, 1), 6),
                "false_downgrade_rate_b5": round(false_downgrades_b5 / max(n_high_risk, 1), 6),
                "false_downgrade_rate_dcri": round(false_downgrades_dcri / max(n_high_risk, 1), 6),
            },
            "sample_packets": packet_evaluations,
        }
