# action.py
# Toy lattice action terms for exploration.
import numpy as np, math

def regge_like_term(edge_curvatures):
    vals = np.array(list(edge_curvatures), dtype=float)
    return float(np.sum(vals * vals))

def info_penalty_term(curv_site, dS_dt, eta: float = 1.0):
    c = np.array(curv_site, dtype=float)
    ds = np.full_like(c, float(dS_dt)) if np.ndim(dS_dt) == 0 else np.array(dS_dt, dtype=float)
    return float(np.sum((c + eta * ds) ** 2))

def lattice_action(edge_curvatures, curv_site, dS_dt, GN: float = 1.0, lam: float = 1.0, eta: float = 1.0):
    Sregge = regge_like_term(edge_curvatures)
    Sinfo = info_penalty_term(curv_site, dS_dt, eta=eta)
    return (1.0 / (16.0 * math.pi * GN)) * Sregge + lam * Sinfo
