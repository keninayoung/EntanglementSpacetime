# graph_ops.py
# Graph utilities for discrete divergence and Bianchi-style residuals.
import numpy as np
from typing import List, Tuple

def bianchi_residual(G_site: np.ndarray, edges: List[Tuple[int, int]], weights: np.ndarray, norm: str = "l2") -> float:
    N = G_site.shape[0]
    neigh = [[] for _ in range(N)]; w_by_edge = [[] for _ in range(N)]
    for e_idx, (i, j) in enumerate(edges):
        w = float(weights[e_idx])
        neigh[i].append(j); w_by_edge[i].append(w)
        neigh[j].append(i); w_by_edge[j].append(w)
    div = np.zeros((N,), dtype=float)
    for i in range(N):
        s = 0.0
        for j, w in zip(neigh[i], w_by_edge[i]):
            s += w * (G_site[j] - G_site[i])
        div[i] = s
    if norm == "linf":
        num = np.linalg.norm(div, ord=np.inf); den = np.linalg.norm(G_site, ord=np.inf)
    else:
        num = np.linalg.norm(div, ord=2); den = np.linalg.norm(G_site, ord=2)
    return float(num if den == 0.0 else num / den)
