# R3: Scaling and summary using consolidated curvature_lattice.csv
# Reads spacetime_outputs/curvature_lattice.csv written as:
#   rows = sites named "site_i", columns = "Step 0", "Step 1", ...
# Plots mean site curvature at t=0 vs N (single point if one run),
# and also a small time trace of mean curvature for context.

import os
import sys
import argparse
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def run(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    root = cfg["paths"]["spacetime_outputs_dir"]
    out_dir = cfg["paths"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    grid_csv = os.path.join(root, "curvature_lattice.csv")
    if not os.path.isfile(grid_csv):
        print("[R3] Missing file:", grid_csv)
        print("[R3] Generate it by running your simulation.")
        sys.exit(1)

    df = pd.read_csv(grid_csv, index_col=0)
    # Columns are "Step 0", "Step 1", ...
    step_cols = [c for c in df.columns if c.lower().startswith("step")]
    if not step_cols:
        print("[R3] No step columns found. Columns were:", list(df.columns))
        sys.exit(1)

    # N = number of sites
    N = df.shape[0]
    # Mean curvature at t=0 across sites
    if "Step 0" in df.columns:
        mean0 = float(df["Step 0"].mean())
    else:
        mean0 = float(df[step_cols[0]].mean())

    # Plot scaling point (single run -> single point)
    plt.figure()
    plt.plot([N], [mean0], marker="o")
    plt.xlabel("System size N (number of sites)")
    plt.ylabel("Mean site curvature at t=0")
    plt.title("R3: Curvature vs system size (single run)")
    plt.grid(True)
    out_png1 = os.path.join(out_dir, "R3_curvature_vs_size.png")
    plt.savefig(out_png1, dpi=180, bbox_inches="tight")
    print("[R3] Wrote", out_png1)

    # Also show mean curvature over time for this run
    step_indices = []
    step_means = []
    for c in step_cols:
        try:
            idx = int(c.split()[-1])
        except Exception:
            idx = len(step_indices)
        step_indices.append(idx)
        step_means.append(float(df[c].mean()))
    order = np.argsort(np.array(step_indices))
    t_ordered = np.array(step_indices)[order]
    m_ordered = np.array(step_means)[order]

    plt.figure()
    plt.plot(t_ordered, m_ordered, marker="o")
    plt.xlabel("Time index")
    plt.ylabel("Mean site curvature")
    plt.title("R3: Mean curvature over time (single run)")
    plt.grid(True)
    out_png2 = os.path.join(out_dir, "R3_mean_curvature_over_time.png")
    plt.savefig(out_png2, dpi=180, bbox_inches="tight")
    print("[R3] Wrote", out_png2)


def main():
    ap = argparse.ArgumentParser()
    default_conf = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
    ap.add_argument("--conf", default=default_conf)
    args = ap.parse_args()
    run(args.conf)


if __name__ == "__main__":
    main()
