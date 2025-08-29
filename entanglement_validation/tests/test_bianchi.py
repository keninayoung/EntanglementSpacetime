# Smoke test: constant field yields zero divergence residual.
import numpy as np
from utils.graph_ops import bianchi_residual

def test_bianchi_on_constant_field():
    N = 5
    G = np.ones((N,), dtype=float)
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    w = np.ones((len(edges),), dtype=float)
    res = bianchi_residual(G, edges, w, norm="l2")
    assert res == 0.0
