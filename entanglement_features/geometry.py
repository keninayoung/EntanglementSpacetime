# geometry.py
# Discrete geometric proxies built from MI.
from typing import List, Tuple, Dict

def kappa_from_mi(Iij: float) -> float:
    # Edge curvature proxy kappa_ij = -I_ij
    return -float(Iij)

def nodewise_einstein_tensor(alpha: float,
                             edges: List[Tuple[int, int]],
                             kappas: Dict[Tuple[int, int], float]) -> Dict[int, float]:
    # Nodewise Einstein-like tensor G(i) = alpha * mean_j kappa_ij over neighbors
    from collections import defaultdict
    neigh = defaultdict(list)
    for (i, j) in edges:
        kij = kappas.get((i, j), kappas.get((j, i), 0.0))
        neigh[i].append(kij); neigh[j].append(kij)
    G = {}
    for i, vals in neigh.items():
        G[i] = 0.0 if not vals else float(alpha) * (sum(vals) / len(vals))
    return G

def edge_weights_from_distance(edges: List[Tuple[int, int]],
                               dists: Dict[Tuple[int, int], float],
                               eps: float = 1e-9) -> List[float]:
    # Build edge weights w_ij = 1 / d_ij with epsilon floor
    ws = []
    for (i, j) in edges:
        d = dists.get((i, j), dists.get((j, i), None))
        if d is None or d <= eps:
            ws.append(1.0)
        else:
            ws.append(1.0 / float(d))
    return ws
