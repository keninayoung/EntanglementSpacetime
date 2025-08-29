# horizon.py
# Horizon mutual information across a bipartition A | A^c.
from typing import Iterable, Tuple, Dict, Set

def horizon_mi_sum(mi_edges: Dict[Tuple[int, int], float], A: Iterable[int]) -> float:
    Aset: Set[int] = set(A)
    total = 0.0
    for (i, j), val in mi_edges.items():
        if (i in Aset) ^ (j in Aset):
            total += float(val)
    return total
