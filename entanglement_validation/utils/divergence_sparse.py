# divergence_sparse.py
# Fast discrete divergence using a CSR matrix.
import numpy as np
from scipy.sparse import csr_matrix

def build_divergence_matrix(i_edges, j_edges, w_ij, n_nodes):
    i_edges = np.asarray(i_edges); j_edges = np.asarray(j_edges); w_ij = np.asarray(w_ij)
    rows = np.concatenate([i_edges, i_edges, j_edges, j_edges])
    cols = np.concatenate([j_edges, i_edges, i_edges, j_edges])
    data = np.concatenate([+w_ij, -w_ij, +w_ij, -w_ij])
    D = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    return D

def bianchi_residual_sparse(G, D, norm="l2"):
    div = D @ G
    if norm == "linf":
        num = np.max(np.abs(div)); den = np.max(np.abs(G))
    else:
        num = np.linalg.norm(div); den = np.linalg.norm(G)
    return float(num if den == 0.0 else num / den)
