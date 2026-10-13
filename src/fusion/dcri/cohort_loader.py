"""
FusionMedAI - Phase C11.6: Frozen Decision Packet Loader
Loads the frozen cohort of N=500 ControlledDecisionPackets from Phase C11.5 / C11.6.
Strict Fail-Closed Policy: Requires exact frozen packet manifest on disk with explicit seed and unique IDs; refuses silent fallback.
"""

from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import json

from src.fusion.baselines.decision_packet import ModalityRecord, ControlledDecisionPacket, DecisionPacketError


class CohortManifestError(RuntimeError):
    """Raised when frozen cohort manifest is missing, incomplete, or corrupted."""
    pass


# Authoritative Canonical Redesigned Cohort (Single Authoritative Source of Truth)
CANONICAL_MANIFEST_REL_PATH: str = "experiments/fusion/canonical_redesigned_rerun/01_canonical_redesigned_packet_manifest.json"
CANONICAL_MANIFEST_SHA256: str = "cf877d74e306d6dbc3ac7d9e2485b1c46da4cd64a8e35a82fe68274854dc4cd4"
CANONICAL_CONTENT_STREAM_DIGEST: str = "65d7c53c3238128d34ce3efe4b121d656c9450df7b0eed48fc8cdb4a09090ea2"

# Historical Baseline Decision Cohort (Preserved for Historical Reproducibility; Explicit Opt-In Only)
HISTORICAL_BASELINE_MANIFEST_REL_PATH: str = "experiments/fusion/dcri/packet_manifest.json"
HISTORICAL_BASELINE_MANIFEST_SHA256: str = "1ff37fbeeba33feae7bf14d57d0df4a10df48ac789edeedf9dbd2fa8121382bb"
HISTORICAL_BASELINE_CONTENT_STREAM_DIGEST: str = "491da3d908652915e51f744d03c89c55db776549a8b18d6c32771c12d18e96bd"


def load_frozen_cohort(
    repo_root: Path,
    n_packets: int = 500,
    seed: int = 115,
    manifest_path: Optional[Path] = None,
    expected_sha256: Optional[str] = None,
) -> List[ControlledDecisionPacket]:
    """
    Loads a frozen cohort of ControlledDecisionPackets.
    
    Enforces a strict Fail-Closed policy:
    1. By default (manifest_path is None), resolves to the single authoritative
       Canonical Redesigned Cohort: experiments/fusion/canonical_redesigned_rerun/01_canonical_redesigned_packet_manifest.json.
       Verifies the authoritative SHA-256 (cf877d74...) fail-closed unless expected_sha256 is explicitly specified.
    2. Historical/alternative cohorts (e.g., legacy Baseline Decision Cohort in experiments/fusion/dcri/packet_manifest.json)
       require explicit opt-in by passing manifest_path. Silent fallback to legacy manifests is strictly forbidden.
    3. If expected_sha256 is provided, verifies that the file hash matches before loading.
    4. Validates that the manifest contains an explicit 'seed' field and manifest['seed'] == seed.
    5. Validates that the manifest contains exactly n_packets with strictly unique packet_ids.
    6. If manifest is missing, incomplete, or corrupted, fails closed immediately.
    """
    if manifest_path is None:
        manifest_path = repo_root / CANONICAL_MANIFEST_REL_PATH
        if expected_sha256 is None:
            expected_sha256 = CANONICAL_MANIFEST_SHA256


    if manifest_path is None or not manifest_path.is_file():
        raise CohortManifestError(
            f"Fail-Closed: Frozen packet manifest not found at {manifest_path}. "
            "Silent reconstruction is prohibited for sealed research artifacts."
        )

    if expected_sha256 is not None:
        import hashlib
        h = hashlib.sha256()
        with open(manifest_path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        actual_hash = h.hexdigest()
        if actual_hash != expected_sha256:
            raise CohortManifestError(
                f"Fail-Closed: Manifest SHA-256 mismatch for {manifest_path}.\n"
                f"  Expected: {expected_sha256}\n"
                f"  Actual:   {actual_hash}"
            )

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    except Exception as e:
        raise CohortManifestError(f"Fail-Closed: Failed to read or parse manifest {manifest_path}: {e}")

    # Strict seed presence and value check (no missing seed allowed)
    if "seed" not in manifest:
        raise CohortManifestError(
            f"Fail-Closed: Manifest {manifest_path} is missing mandatory 'seed' field."
        )
    if manifest["seed"] != seed:
        raise CohortManifestError(
            f"Fail-Closed: Manifest seed mismatch: expected {seed}, found {manifest['seed']} in {manifest_path}."
        )

    sample_packets = manifest.get("sample_packets", [])
    if len(sample_packets) != n_packets:
        raise CohortManifestError(
            f"Fail-Closed: Expected exactly {n_packets} packets in {manifest_path}, but found {len(sample_packets)}."
        )

    def _to_mod_record(d: Dict[str, Any]) -> ModalityRecord:
        kwargs = dict(d)
        if "calibrated_probability" in kwargs and isinstance(kwargs["calibrated_probability"], list):
            kwargs["calibrated_probability"] = tuple(float(x) for x in kwargs["calibrated_probability"])
        return ModalityRecord(**kwargs)

    # Reconstruct packets strictly through immutable dataclass validation & verify unique IDs
    packets: List[ControlledDecisionPacket] = []
    seen_ids: Set[str] = set()

    for i, pd in enumerate(sample_packets):
        p_id = pd.get("packet_id")
        if not p_id or p_id in seen_ids:
            raise CohortManifestError(
                f"Fail-Closed: Duplicate or missing packet_id '{p_id}' at index {i} in {manifest_path}."
            )
        seen_ids.add(p_id)

        try:
            packet = ControlledDecisionPacket(
                packet_id=p_id,
                retina=_to_mod_record(pd["retina"]),
                foot=_to_mod_record(pd["foot"]),
                clinical=_to_mod_record(pd["clinical"]),
                seed=pd.get("seed", seed),
                packet_type=pd.get("packet_type", "CONTROLLED_DECISION_PACKET"),
            )
            packets.append(packet)
        except Exception as e:
            raise CohortManifestError(
                f"Fail-Closed: Corrupted packet record at index {i} in {manifest_path}: {e}"
            )

    return packets
