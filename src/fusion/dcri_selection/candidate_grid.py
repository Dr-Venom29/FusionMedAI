"""
FusionMedAI - Phase C11.13: Pre-Specified Candidate Delta Grid
Defines candidate grid items, metadata, and invariant validation.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
from .selection_config import CANDIDATE_DELTA_GRID, DELTA_PROVISIONAL_HISTORICAL


class CandidateGridError(ValueError):
    """Raised when candidate grid definition violates protocol specifications."""
    pass


@dataclass(frozen=True)
class DeltaCandidateItem:
    """
    Immutable representation of a candidate delta parameter.
    
    Attributes:
        candidate_id: Identifier (e.g. 'D00', 'D05', 'D10', ..., 'D100').
        delta: Numeric delta value in [0.0, 1.0].
        name: Descriptive name.
        category: Grid region category ('zero_penalty', 'fine_candidate', 'moderate_penalty', 'strong_penalty').
        description: Methodological rationale for inclusion.
        is_zero: Boolean indicating if delta == 0.0.
        is_provisional_match: Boolean indicating if delta matches historical 0.20.
    """
    candidate_id: str
    delta: float
    name: str
    category: str
    description: str
    is_zero: bool = False
    is_provisional_match: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "delta": self.delta,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "is_zero": self.is_zero,
            "is_provisional_match": self.is_provisional_match,
        }


def _get_category(delta: float) -> str:
    if delta == 0.0:
        return "zero_penalty"
    elif delta <= 0.30:
        return "fine_candidate"
    elif delta <= 0.50:
        return "moderate_penalty"
    else:
        return "strong_penalty"


def _generate_candidate_id(delta: float) -> str:
    val = int(round(delta * 100))
    return f"D{val:02d}"


def get_candidate_grid() -> List[DeltaCandidateItem]:
    """
    Returns the pre-specified 11-point candidate delta grid.
    
    Returns:
        List of DeltaCandidateItem objects.
    """
    items: List[DeltaCandidateItem] = []
    for d in CANDIDATE_DELTA_GRID:
        cid = _generate_candidate_id(d)
        cat = _get_category(d)
        is_zero = (d == 0.0)
        is_prov = (abs(d - DELTA_PROVISIONAL_HISTORICAL) < 1e-6)
        
        if is_zero:
            desc = "Zero-penalty baseline: DCRI strictly equals R_fusion; test uncertainty discount justification."
            name = "Zero Penalty (Base Fusion Risk)"
        elif d <= 0.15:
            desc = f"Conservative fine-resolution penalty (delta = {d:.2f}): mild uncertainty attenuation."
            name = f"Conservative Penalty ({d:.2f})"
        elif d == 0.20:
            desc = "Historical provisional reference operating point (delta = 0.20)."
            name = "Provisional Historical Reference (0.20)"
        elif d <= 0.30:
            desc = f"Moderate fine-resolution penalty (delta = {d:.2f}): intermediate uncertainty discounting."
            name = f"Moderate Penalty ({d:.2f})"
        elif d <= 0.50:
            desc = f"Substantial uncertainty penalty (delta = {d:.2f}): aggressive discounting for high-uncertainty channels."
            name = f"Substantial Penalty ({d:.2f})"
        else:
            desc = f"Strong penalty boundary condition (delta = {d:.2f}): full scale or extreme boundary test."
            name = f"Boundary Penalty ({d:.2f})"
            
        item = DeltaCandidateItem(
            candidate_id=cid,
            delta=d,
            name=name,
            category=cat,
            description=desc,
            is_zero=is_zero,
            is_provisional_match=is_prov,
        )
        items.append(item)
    return items


def validate_candidate_grid(grid: List[DeltaCandidateItem]) -> None:
    """
    Validates that the candidate grid strictly conforms to protocol invariants.
    
    Raises:
        CandidateGridError: If grid length, uniqueness, ordering, or values are invalid.
    """
    if len(grid) != len(CANDIDATE_DELTA_GRID):
        raise CandidateGridError(f"Grid length {len(grid)} does not match expected {len(CANDIDATE_DELTA_GRID)}")
    
    deltas = [item.delta for item in grid]
    ids = [item.candidate_id for item in grid]
    
    if len(set(ids)) != len(ids):
        raise CandidateGridError("Candidate IDs are not strictly unique")
    
    if len(set(deltas)) != len(deltas):
        raise CandidateGridError("Candidate delta values are not strictly unique")
        
    for i, d in enumerate(deltas):
        if not math.isfinite(d):
            raise CandidateGridError(f"Delta {d} is non-finite (NaN or Inf)")
        if d < 0.0 or d > 1.0:
            raise CandidateGridError(f"Delta {d} is outside admissible mathematical bounds [0.0, 1.0]")
        if i > 0 and d <= deltas[i-1]:
            raise CandidateGridError(f"Delta grid must be strictly monotonically increasing: {deltas[i-1]} >= {d}")
            
    if deltas[0] != 0.0:
        raise CandidateGridError("First candidate delta must be strictly 0.0 (zero penalty)")
    if deltas[-1] != 1.0:
        raise CandidateGridError("Last candidate delta must be strictly 1.0 (full penalty boundary)")
