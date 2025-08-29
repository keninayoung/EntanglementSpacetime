# etensor.py
# CSV pipeline: MI -> curvature, Einstein-like tensor, optional page curve.
import csv, glob, os
from typing import Dict, Tuple, List
from .units import mi_to_distance, UnitsConfig
from .geometry import kappa_from_mi, nodewise_einstein_tensor, edge_weights_from_distance
from .horizon import horizon_mi_sum

def load_mi_csv(path: str):
    # Expect columns: t,i,j,Iij (t optional)
    edges = []; mi_map: Dict[Tuple[int, int], float] = {}; t_val = None
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            i = int(row["i"]); j = int(row["j"])
            if i == j: continue
            I = float(row["Iij"])
            a, b = (i, j) if i < j else (j, i)
            if (a, b) not in mi_map:
                edges.append((a, b)); mi_map[(a, b)] = I
            if "t" in row and row["t"]:
                t_val = float(row["t"])
    if t_val is None:
        try:
            base = os.path.splitext(os.path.basename(path))[0]
            t_val = float(base.split("t")[-1])
        except Exception:
            t_val = 0.0
    return t_val, edges, mi_map

def write_curvature_csv(out_path: str, edges: List[Tuple[int, int]], kappas: Dict[Tuple[int, int], float]):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["i", "j", "kappa"])
        for (i, j) in edges:
            kij = kappas.get((i, j), kappas.get((j, i), 0.0))
            wr.writerow([i, j, kij])

def write_einstein_csv(out_path: str, G: Dict[int, float], edges: List[Tuple[int, int]], weights: List[float]):
    from collections import defaultdict
    neigh = defaultdict(list); wghs = defaultdict(list)
    for (i, j), w in zip(edges, weights):
        neigh[i].append(j); wghs[i].append(w)
        neigh[j].append(i); wghs[j].append(w)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["node_id", "G_value", "neighbors", "weights"])
        for nid, gval in G.items():
            nlist = ";".join(str(x) for x in neigh.get(nid, []))
            wlist = ";".join(f"{x:.8g}" for x in wghs.get(nid, []))
            wr.writerow([nid, f"{gval:.8g}", nlist, wlist])

def write_horizon_csv(out_path: str, t_vals: List[float], horizon_vals: List[float]):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f); wr.writerow(["t", "horizon_mi"])
        for t, v in zip(t_vals, horizon_vals):
            wr.writerow([t, v])

def process_folder(mi_glob: str, out_dir: str, alpha: float = 1.0,
                   units: UnitsConfig = UnitsConfig(), horizon_A: List[int] = None):
    os.makedirs(out_dir, exist_ok=True)
    t_series = []; horizon_series = []
    for path in sorted(glob.glob(mi_glob)):
        t, edges, mi_map = load_mi_csv(path)
        kappas = {e: kappa_from_mi(mi_map[e]) for e in edges}
        dists = {e: mi_to_distance(mi_map[e], units.ell0) for e in edges}
        weights = edge_weights_from_distance(edges, dists)
        G_site = nodewise_einstein_tensor(alpha, edges, kappas)
        write_curvature_csv(os.path.join(out_dir, f"curvature_t{t}.csv"), edges, kappas)
        write_einstein_csv(os.path.join(out_dir, f"einstein_tensor_t{t}.csv"), G_site, edges, weights)
        if horizon_A is not None:
            t_series.append(t); horizon_series.append(horizon_mi_sum(mi_map, horizon_A))
    if horizon_A is not None and t_series:
        write_horizon_csv(os.path.join(out_dir, "pagecurve.csv"), t_series, horizon_series)
