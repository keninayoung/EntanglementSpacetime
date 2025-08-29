# units.py
# Dimension mapping utilities (ASCII only).
from dataclasses import dataclass
import math
from typing import Iterable

@dataclass(frozen=True)
class UnitsConfig:
    # ell0: length scale for MI->distance mapping
    # chi: optional stiffness scale for action-like terms
    ell0: float = 1.0
    chi: float = 1.0

def mi_to_distance(Iij: float, ell0: float, eps: float = 1e-12) -> float:
    # Map dimensionless MI to a distance-like quantity: d_ij = ell0 * (-log(Iij))
    val = max(Iij, eps)
    return ell0 * (-math.log(val))

def effective_entanglement_tensor_scalar(I_neighbors: Iterable[float]) -> float:
    # Site-level scalar proxy for E(i) by averaging neighbor MI
    I_neighbors = list(I_neighbors)
    if not I_neighbors:
        return 0.0
    return float(sum(I_neighbors) / len(I_neighbors))
