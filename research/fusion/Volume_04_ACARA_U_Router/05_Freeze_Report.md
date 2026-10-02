# 05 ACARA-U Router Freeze Report & Handoff — Phase C11.4

## 1. Executive Summary & Router Architecture Lock

Phase C11.4 concludes with the formal freeze of the **ACARA-U v2 Dynamic Multimodal Router**:
- Implementation: [`src/fusion/router/acarau_router.py`](../../../src/fusion/router/acarau_router.py)
- Verification: [`verification/fusion/router/verify_router.py`](../../../verification/fusion/router/verify_router.py) (**`18 / 18 GATES PASSED`**)

---

## 2. Frozen Baseline Configurations (B2–B6)

```json
{
  "B2_Uniform": {"alpha": 0.0, "beta": 0.0, "gamma": 0.0, "eta": 0.0},
  "B3_Confidence": {"alpha": 1.0, "beta": 0.0, "gamma": 0.0, "eta": 0.0},
  "B4_Conf_Rel": {"alpha": 1.0, "beta": 1.0, "gamma": 0.0, "eta": 0.0},
  "B5_Conf_Rel_Unc": {"alpha": 1.0, "beta": 1.0, "gamma": 1.0, "eta": 0.0},
  "B6_Full_ACARA_U": {"alpha": 1.0, "beta": 1.0, "gamma": 1.0, "eta": 1.0}
}
```

---

## 3. Downstream Handoff to Phase C11.5 (Multimodal Decision Fusion & DCRI)

With dynamic weights $w_i \ge 0$ ($\sum_{i \in \mathcal{A}} w_i = 1.0$) mathematically verified, Phase C11.5 can now proceed to implement:
1. **Fused Risk Aggregation**: $R_{\text{fusion}} = \sum_{i \in \mathcal{A}} w_i r_i \in [0.0, 1.0]$.
2. **Dynamic Clinical Risk Index**: $DCRI = R_{\text{fusion}} - \delta \sum_{i \in \mathcal{A}} U_i$.
3. **Consensus & Conflict Metrics**: Inter-modality risk divergence and discordant pair flagging.

---

## 4. C11.5 Scientific Boundary

Because the Retina, Foot Ulcer, and Clinical datasets do not contain paired observations for the same patients, $R_{\text{fusion}}$ and DCRI will be evaluated as **decision-level derived indices and behavioral fusion outputs**, not as clinically validated probabilities or patient-level ground-truth outcomes.

C11.5 will therefore emphasize baseline comparison, modality ablation, missing-modality robustness, uncertainty sensitivity, quality sensitivity, and conflict/discordance behavior.
