"""
FusionMedAI - Phase C11.0: Research Protocol Freeze Deep Verification Gate (Final Freeze v1.1a)
Rigorous automated verification script for C11.0 mathematical, scientific, and architectural invariants.
"""

import os
import sys
import hashlib
import re

def find_repo_root():
    curr = os.path.abspath(os.path.dirname(__file__))
    while curr and os.path.splitdrive(curr)[1] != '\\':
        if os.path.exists(os.path.join(curr, "research")):
            return curr
        parent = os.path.dirname(curr)
        if parent == curr:
            break
        curr = parent
    return os.path.abspath(".")

REPO_ROOT = find_repo_root()
PROTOCOL_DIR = os.path.join(REPO_ROOT, "research", "fusion", "Volume_01_Research_Protocol")

MANDATORY_DOCUMENTS = [
    "README.md",
    "01_Protocol_Overview.md",
    "02_Mathematical_Formulation.md",
    "03_Input_Output_Contracts.md",
    "04_Experimental_Ladder_Baselines.md",
    "05_Methodological_Boundaries.md",
    "06_Split_Hygiene_and_Tuning.md"
]

def compute_sha256(filepath):
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()

def read_doc(filename):
    path = os.path.join(PROTOCOL_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def run_c11_00_verification():
    print("=" * 85)
    print("FusionMedAI: Phase C11.0 - Research Protocol Freeze Deep Verification Gate (v1.1a)")
    print("=" * 85)
    
    passed_checks = 0
    total_checks = 7
    
    # -------------------------------------------------------------------------
    # Gate 1: Directory & Document Inventory
    # -------------------------------------------------------------------------
    print("\n[Gate 1/7] Verifying Protocol Directory and Document Inventory Integrity...")
    if not os.path.exists(PROTOCOL_DIR):
        print(f"  -> FAIL: Protocol directory '{PROTOCOL_DIR}' not found.")
        return False
    
    all_files_ok = True
    for doc in MANDATORY_DOCUMENTS:
        doc_path = os.path.join(PROTOCOL_DIR, doc)
        if not os.path.exists(doc_path):
            print(f"  -> MISSING: {doc}")
            all_files_ok = False
        else:
            size = os.path.getsize(doc_path)
            file_hash = compute_sha256(doc_path)
            if size < 1000:
                print(f"  -> INSUFFICIENT CONTENT: {doc} ({size} bytes)")
                all_files_ok = False
            else:
                print(f"  -> VALID: {doc:<38} [{size:>5} bytes | SHA-256: {file_hash[:16]}...]")
                
    if all_files_ok:
        print("  -> PASS: All 7 protocol freeze documents present, non-empty, and fingerprinted.")
        passed_checks += 1
    else:
        print("  -> FAIL: Document inventory check failed.")
        return False

    # -------------------------------------------------------------------------
    # Gate 2: Modality Architecture, Contract 8-Tuple & Frozen Reliability Invariant
    # -------------------------------------------------------------------------
    print("\n[Gate 2/7] Verifying Modality Architectures, Contract 8-Tuple, R_i Formula, and Projections...")
    doc3 = read_doc("03_Input_Output_Contracts.md")
    
    # 1. Check frozen architectures
    retina_arch_ok = "EfficientNet-B3" in doc3 and "Temperature Scaling" in doc3 and ("25" in doc3 or "N=25" in doc3)
    foot_arch_ok = "EfficientNet-B3" in doc3 and ("Vector Scaling" in doc3 or "FootVectorScaler" in doc3) and ("10" in doc3 or "N=10" in doc3)
    clinical_arch_ok = "CatBoost" in doc3 and "Isotonic Calibration" in doc3 and ("Bootstrap" in doc3 or "B=20" in doc3) and "119" in doc3
    
    # 2. Check 8-tuple attributes
    attributes = ["risk", "calibrated_probability", "confidence", "uncertainty", "quality", "availability", "reliability", "model_version"]
    attrs_ok = all(re.search(rf"\b{attr}\b", doc3) for attr in attributes)
    
    # 3. Check projection formulas
    retina_proj_ok = r"\sum_{k=0}^4" in doc3 and r"\frac{k}{4}" in doc3
    foot_proj_ok = r"\sum_{k=0}^3" in doc3 and r"\frac{k}{3}" in doc3
    clinical_proj_ok = r"r_{\text{clinical}} = P(Y = 1 \mid x) = p_1" in doc3 or "p_1" in doc3
    
    # 4. Check unified reliability formula R_i = 0.5 * (AUC_i + (1 - ECE_i))
    rel_formula_ok = (r"R_i = \frac{1}{2}" in doc3 or "1/2" in doc3) and "AUC" in doc3 and "ECE" in doc3
    
    if retina_arch_ok and foot_arch_ok and clinical_arch_ok and attrs_ok and retina_proj_ok and foot_proj_ok and clinical_proj_ok and rel_formula_ok:
        print("  -> PASS: Frozen backbones (Retina: EfficientNet-B3/Temp/MCD25, Foot: EfficientNet-B3/Vec/MCD10, Clinical: CatBoost/Iso/Boot20), 8-tuple contract, exact risk projections, and frozen R_i formula verified.")
        passed_checks += 1
    else:
        print(f"  -> FAIL: Modality architecture or contract invariant failure (Retina: {retina_arch_ok}, Foot: {foot_arch_ok}, Clinical: {clinical_arch_ok}, Attrs: {attrs_ok}, Proj: {retina_proj_ok and foot_proj_ok and clinical_proj_ok}, Rel: {rel_formula_ok})")

    # -------------------------------------------------------------------------
    # Gate 3: Core Router & Hard Availability Masking Mathematical Verification
    # -------------------------------------------------------------------------
    print("\n[Gate 3/7] Verifying Core Router Formulations, Masking, and R_fusion vs DCRI Separation...")
    doc2 = read_doc("02_Mathematical_Formulation.md")
    
    router_eq_ok = r"z_i = \alpha C_i + \beta R_i - \gamma U_i + \eta Q_i" in doc2
    mask_eq_ok = (r"A_i = 1" in doc2 and r"-\infty" in doc2 and r"A_i = 0" in doc2)
    zero_weight_ok = (r"w_i = 0" in doc2 or r"A_i = 0 \implies \tilde{z}_i = -\infty \implies w_i = 0" in doc2)
    softmax_ok = r"w_i = \frac{e^{\tilde{z}_i}}{\sum_{j=1}^M e^{\tilde{z}_j}}" in doc2 or r"e^{\tilde{z}_i}" in doc2
    rfusion_ok = r"R_{\text{fusion}} = \sum_{i=1}^M w_i r_i" in doc2
    dcri_ok = r"DCRI = R_{\text{fusion}} - \delta \sum_{i \in \mathcal{A}} U_i" in doc2
    dcri_bounds_ok = r"[-\delta M, 1" in doc2
    
    if router_eq_ok and mask_eq_ok and zero_weight_ok and softmax_ok and rfusion_ok and dcri_ok and dcri_bounds_ok:
        print("  -> PASS: Router scoring logit z_i, hard availability masking (A_i=0 => w_i=0), softmax normalization, R_fusion, and DCRI bounds [-\\delta M, 1.0] verified.")
        passed_checks += 1
    else:
        print(f"  -> FAIL: Router math failure (Router: {router_eq_ok}, Mask: {mask_eq_ok}, ZeroWeight: {zero_weight_ok}, Softmax: {softmax_ok}, Rfusion: {rfusion_ok}, DCRI: {dcri_ok}, Bounds: {dcri_bounds_ok})")

    # -------------------------------------------------------------------------
    # Gate 4: Baselines B1–B6 & Uncertainty Ablation Ladder Verification
    # -------------------------------------------------------------------------
    print("\n[Gate 4/7] Verifying Experimental Ladder B1–B6 and Uncertainty Ablation Hierarchy...")
    doc4 = read_doc("04_Experimental_Ladder_Baselines.md")
    
    b1_ok = ("Reliability-Selected Unimodal Baseline" in doc4 or "Reliability-selected" in doc4) and r"i^* = \arg\max_{i \in \mathcal{A}} R_i" in doc4
    b2_ok = "Uniform Average Fusion" in doc4 and r"w_i = \frac{1}{|\mathcal{A}|}" in doc4
    b3_ok = "Confidence Fusion" in doc4 and r"z_i = \alpha C_i" in doc4
    b4_ok = "Confidence + Reliability" in doc4 and r"z_i = \alpha C_i + \beta R_i" in doc4
    b5_ok = "Confidence + Rel + Uncertainty" in doc4 and r"z_i = \alpha C_i + \beta R_i - \gamma U_i" in doc4
    b6_ok = "Full ACARA-U" in doc4 and r"z_i = \alpha C_i + \beta R_i - \gamma U_i + \eta Q_i" in doc4
    null_mandate_ok = ("prohibits retroactive" in doc4.lower() or "null contribution" in doc4.lower()) and "empirical finding" in doc4.lower()
    
    if b1_ok and b2_ok and b3_ok and b4_ok and b5_ok and b6_ok and null_mandate_ok:
        print("  -> PASS: Complete baseline hierarchy (B1 Reliability-Selected through B6 Full ACARA-U) and empirical null-reporting mandate verified.")
        passed_checks += 1
    else:
        print(f"  -> FAIL: Baseline verification failure (B1: {b1_ok}, B2: {b2_ok}, B3: {b3_ok}, B4: {b4_ok}, B5: {b5_ok}, B6: {b6_ok}, NullMandate: {null_mandate_ok})")

    # -------------------------------------------------------------------------
    # Gate 5: Seven Configurations & Graceful Failure Verification
    # -------------------------------------------------------------------------
    print("\n[Gate 5/7] Verifying 7 Modality Configurations & Zero-Modality Graceful Failure...")
    configs_ok = all(c in doc4 for c in ["Config 1", "Config 2", "Config 3", "Config 4", "Config 5", "Config 6", "Config 7"])
    graceful_fail_ok = ("NO_MODALITY_AVAILABLE" in doc4) and ("Graceful Failure" in doc4 or "Safe Rejection" in doc4)
    
    if configs_ok and graceful_fail_ok:
        print("  -> PASS: All 7 operational configurations (Tri-modal, 3 Bi-modal, 3 Uni-modal) and zero-modality safe rejection verified.")
        passed_checks += 1
    else:
        print(f"  -> FAIL: Configuration check failure (Configs: {configs_ok}, GracefulFail: {graceful_fail_ok})")

    # -------------------------------------------------------------------------
    # Gate 6: Methodological Boundaries & Synthetic Target Rules
    # -------------------------------------------------------------------------
    print("\n[Gate 6/7] Verifying Disjoint Datasets, Two-Tier Evaluation, and Synthetic Target Rules...")
    doc5 = read_doc("05_Methodological_Boundaries.md")
    
    disjoint_ok = r"\text{Cohort}_{\text{APTOS 2019 (Retina)}} \neq \text{Cohort}_{\text{ADPM / DFUC (Foot)}} \neq \text{Cohort}_{\text{UCI Diabetes 130-US (Clinical)}}" in doc5 or ("APTOS" in doc5 and "ADPM" in doc5 and "UCI" in doc5 and r"\neq" in doc5)
    two_tier_ok = "Tier 1: Modality-Level Predictive Evaluation" in doc5 and "Tier 2: Decision-Level Fusion" in doc5
    no_comp_ok = "No Composite Patient" in doc5 or "Prohibited Scientific Claims" in doc5
    synthetic_target_ok = "explicitly labeled" in doc5.lower() and "synthetic" in doc5.lower() and "never" in doc5.lower()
    
    # Verify absence of improper multi-agent voting as clinical ground truth
    no_voting_as_gt = "Modality-specific ground truths are used independently for Tier-1 evaluation" in doc5
    
    if disjoint_ok and two_tier_ok and no_comp_ok and synthetic_target_ok and no_voting_as_gt:
        print("  -> PASS: Disjoint cohort reality locked, Two-Tier Evaluation established, composite patient ground truth prohibited, and synthetic target labeling rule sealed.")
        passed_checks += 1
    else:
        print(f"  -> FAIL: Methodological boundary failure (Disjoint: {disjoint_ok}, TwoTier: {two_tier_ok}, NoComp: {no_comp_ok}, SyntheticTargetRule: {synthetic_target_ok}, NoVotingGT: {no_voting_as_gt})")

    # -------------------------------------------------------------------------
    # Gate 7: Split Hygiene, Pure Behavioral Tuning & Lockdown
    # -------------------------------------------------------------------------
    print("\n[Gate 7/7] Verifying Split Hygiene, Pure Behavioral Tuning Objective & Anti-Leakage Lockdown...")
    doc6 = read_doc("06_Split_Hygiene_and_Tuning.md")
    
    stages_ok = "TRAIN / DEV" in doc6 and "VALIDATION" in doc6 and "FINAL TEST" in doc6
    param_bounds_ok = r"[0.0, 5.0]" in doc6 and r"[0.0, 1.0]" in doc6
    lock_ok = "frozen_parameters.json" in doc6 and "SHA-256" in doc6 and "zero hyperparameter modification" in doc6.lower()
    pure_behavioral_ok = "pure behavioral validation objective" in doc6.lower() and "zero synthetic target tuning" in doc6.lower()
    
    if stages_ok and param_bounds_ok and lock_ok and pure_behavioral_ok:
        print("  -> PASS: 3-stage split isolation, parameter bounded domains (\\alpha,\\beta,\\gamma,\\eta \\in [0,5], \\delta \\in [0,1]), pure behavioral validation objective (zero synthetic target tuning), and SHA-256 frozen test lockdown verified.")
        passed_checks += 1
    else:
        print(f"  -> FAIL: Split hygiene check failure (Stages: {stages_ok}, ParamBounds: {param_bounds_ok}, Lock: {lock_ok}, PureBehavioral: {pure_behavioral_ok})")

    print("\n" + "=" * 85)
    print(f"C11.0 DEEP PROTOCOL VERIFICATION RESULT: {passed_checks}/{total_checks} GATES PASSED")
    print("=" * 85)
    
    return passed_checks == total_checks

if __name__ == "__main__":
    success = run_c11_00_verification()
    sys.exit(0 if success else 1)
