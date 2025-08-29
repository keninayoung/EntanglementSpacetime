# R2: Bianchi-type residual over time (matrix mode for consolidated CSVs).
# Reads spacetime_outputs/einstein_tensor.csv written as:
#   rows = time steps, columns = node ids "0","1",...,"N-1"
# If the number of columns is a perfect square, uses a 4-neighbor LxL grid.
# Otherwise falls back to a simple chain graph.
# Output: validation_outputs/R2_bianchi_residual.png

import os
import sys
import argparse
import math
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from entanglement_validation.utils.graph_ops import bianchi_residual


def infer_grid_edges(n_nodes):
    L = int(round(math.sqrt(n_nodes)))
    if L * L != n_nodes:
        return None
    edges = []
    def idx(x, y): return y * L + x
    for y in range(L):
        for x in range(L):
            i = idx(x, y)
            if x + 1 < L:
                edges.append((i, idx(x + 1, y)))
            if y + 1 < L:
                edges.append((i, idx(x, y + 1)))
    return edges


def chain_edges(n_nodes):
    return [(i, i + 1) for i in range(n_nodes - 1)]


def run(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    out_dir = cfg["paths"]["out_dir"]
    root = cfg["paths"]["spacetime_outputs_dir"]
    os.makedirs(out_dir, exist_ok=True)

    matrix_csv = os.path.join(root, "einstein_tensor.csv")
    if not os.path.isfile(matrix_csv):
        print("[R2] Missing file:", matrix_csv)
        print("[R2] Generate it by running your simulation or point config to the right folder.")
        sys.exit(1)

    df = pd.read_csv(matrix_csv)  # rows=time, cols=nodes
    # Columns are expected to be "0","1",...,"N-1"
    try:
        node_cols = [int(c) for c in df.columns]
    except Exception:
        print("[R2] Columns must be integer-named (0,1,2,...). Found:", list(df.columns))
        sys.exit(1)

    n_nodes = len(node_cols)
    edges = infer_grid_edges(n_nodes)
    if edges is None:
        print("[R2] Node count {} is not a perfect square. Using chain graph.".format(n_nodes))
        edges = chain_edges(n_nodes)
    w = np.ones((len(edges),), dtype=float)

    residuals = []
    times = []
    for t_idx in range(len(df)):
        G_row = df.iloc[t_idx].to_numpy(dtype=float)
        res = bianchi_residual(G_row, edges, w, norm=cfg["bianchi"]["norm"])
        residuals.append(res)
        times.append(t_idx)

    plt.figure()
    plt.plot(times, residuals, marker="o")
    plt.xlabel("Time index")
    plt.ylabel("Bianchi residual ||div G|| / ||G||")
    plt.title("R2: Residual over time (matrix mode)")
    plt.grid(True)
    out_png = os.path.join(out_dir, "R2_bianchi_residual.png")
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    print("[R2] Wrote", out_png)


def main():
    ap = argparse.ArgumentParser()
    default_conf = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
    ap.add_argument("--conf", default=default_conf)
    args = ap.parse_args()
    run(args.conf)


if __name__ == "__main__":
    main()
