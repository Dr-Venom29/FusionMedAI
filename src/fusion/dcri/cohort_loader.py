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


def load_frozen_cohort(
    repo_root: Path,
    n_packets: int = 500,
    seed: int = 115,
) -> List[ControlledDecisionPacket]:
    """
    Loads the frozen cohort of N=500 ControlledDecisionPackets.
    
    Enforces a strict Fail-Closed policy:
    1. Looks for frozen packet_manifest.json in experiments/fusion/dcri/ or experiments/fusion/baseline_comparison/.
    2. Validates that the manifest contains an explicit 'seed' field and manifest['seed'] == seed (seed=115).
    3. Validates that the manifest contains exactly n_packets with strictly unique packet_ids.
    4. If manifest is missing, incomplete, or corrupted, fails closed immediately.
    
    Args:
        repo_root: Path to repository root.
        n_packets: Expected number of packets (default 500).
        seed: Expected random seed (default 115).
        
    Returns:
        List of exactly n_packets ControlledDecisionPacket objects.
        
    Raises:
        CohortManifestError: If frozen manifest is missing, incomplete, or corrupted.
    """
    candidates = [
        repo_root / "experiments" / "fusion" / "dcri" / "packet_manifest.json",
        repo_root / "experiments" / "fusion" / "baseline_comparison" / "packet_manifest.json",
    ]

    manifest_path: Optional[Path] = None
    for p in candidates:
        if p.is_file():
            manifest_path = p
            break

    if manifest_path is None:
        raise CohortManifestError(
            f"Fail-Closed: Frozen packet manifest not found in any candidate locations: {candidates}. "
            "Silent reconstruction is prohibited for sealed research artifacts."
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
                retina=ModalityRecord(**pd["retina"]),
                foot=ModalityRecord(**pd["foot"]),
                clinical=ModalityRecord(**pd["clinical"]),
                seed=pd.get("seed", seed),
                packet_type=pd.get("packet_type", "CONTROLLED_DECISION_PACKET"),
            )
            packets.append(packet)
        except Exception as e:
            raise CohortManifestError(
                f"Fail-Closed: Corrupted packet record at index {i} in {manifest_path}: {e}"
            )

    return packets
