# etensor_fast.py
# Vectorized MI -> features using numpy.
import numpy as np

def fast_features_from_mi(i_edges: np.ndarray, j_edges: np.ndarray, Iij: np.ndarray,
                          n_nodes: int, alpha: float = 1.0, ell0: float = 1.0, eps: float = 1e-12):
    kappa = -Iij
    I_safe = np.maximum(Iij, eps)
    d_ij = ell0 * (-np.log(I_safe))
    w_ij = np.where(d_ij > 1e-9, 1.0 / d_ij, 1.0)
    kappa_sum = np.bincount(i_edges, weights=kappa, minlength=n_nodes) +                 np.bincount(j_edges, weights=kappa, minlength=n_nodes)
    degree = np.bincount(i_edges, minlength=n_nodes) + np.bincount(j_edges, minlength=n_nodes)
    degree = np.maximum(degree, 1)
    G = alpha * (kappa_sum / degree)
    return kappa, d_ij, w_ij, G
