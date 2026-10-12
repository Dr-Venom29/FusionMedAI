# Chapter 09: Negative Control Fault Injection

## 1. Objective of Negative Control ($A6$)

A critical requirement in confirmatory software engineering and research reproducibility is verifying that the testing infrastructure and acceptance verifier possess sufficient sensitivity to detect and reject non-compliant candidate architectures.

Candidate $A6$ was designed as a **negative control fault injection experiment** to test the safety invariant enforcement of Gate C16-04.

---

## 2. Fault Injection Specification

### 2.1 Fault Mechanism
In Candidate $A6$, the router executes standard reference logit calculations, but a malicious mutation is injected post-normalization under a bi-modal regime ($\mathcal{A} = \{\text{retina}, \text{foot}\}$, with `clinical` missing):

$$\text{Simplex Check:} \quad \sum_{i} w_i = 1.000000 \quad (\text{Preserved})$$
$$\text{Availability Mask Violation:} \quad w_{\text{clinical}} \leftarrow 0.100000 \neq 0.000000$$
$$w_{\text{retina}} \leftarrow 0.90 \times w_{\text{retina}}, \quad w_{\text{foot}} \leftarrow 0.90 \times w_{\text{foot}}$$

This simulates a stealthy software regression or corruption where weight leaks into an unavailable modality while maintaining sum-to-one simplex normalization.

---

## 3. Experimental Execution & Verifier Audit

The negative control was evaluated across all $N=500$ canonical packets:

| Metric | Target / Expected | Observed in A6 Execution | Verification Outcome |
| :--- | :---: | :---: | :---: |
| Evaluated Packets | $500$ | $500$ | Complete execution |
| Active Modalities | `retina`, `foot` | `retina`, `foot` | Fixed RF Regime |
| Inactive Modality | `clinical` | `clinical` | Target of leak |
| Injected Weight $w_{\text{clinical}}$ | $> 0.0$ | $0.100000$ | Leak present |
| Simplex Violations | $0$ | $0$ | Masked defect |
| **Mask Violations Detected** | **500** | **500** | **100.0% Detection** |
| Verifier Decision | **REJECT** | **REJECTED_AS_INVALID_CANDIDATE** | **Gate C16-04 PASSED** |

### Safety Gate Audit:
The automated verifier detected that $w_{\text{clinical}} > 10^{-6}$ for an inactive channel on every single packet, flagged the invariant breach, and formally issued:
```json
{
  "variant_id": "A6",
  "purpose": "Verify rejection of availability-mask violations",
  "test_regime": "RF",
  "expected_inactive_modality": "clinical",
  "forced_violation_count": 500,
  "total_test_packets": 500,
  "verifier_verdict": "REJECTED_AS_INVALID_CANDIDATE",
  "safety_gate_passed": true
}
```

---

## 4. Scientific Conclusion

The $100\%$ detection rate of Candidate $A6$ confirms that:
1. The acceptance test harness actively verifies availability mask constraints rather than trivially passing all inputs.
2. The verifier demonstrates robust detection and automated rejection of the specific injected post-normalization channel leakage failure mode under the evaluated bi-modal (RF) test fixture. While detecting this targeted synthetic fault demonstrates active verifier sensitivity, comprehensive defense against arbitrary masking defects in real-world deployment requires ongoing runtime monitoring and prospective verification.
3. Gate C16-04 is formally certified.
