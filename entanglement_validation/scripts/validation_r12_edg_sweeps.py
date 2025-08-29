#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# R12: EDG parameter sweeps
#
# Purpose
#   Sweep EDG parameters (epsilon now; L_q placeholder) and measure how the
#   extrapolated perihelion advance A0 depends on them. This reuses:
#     - R6 to generate convergence CSVs for each epsilon (with filename suffix)
#     - R8 to compute A0 per integrator and consensus per planet
#   Then aggregates results across the sweep into summary CSVs and simple plots.
#
# Outputs (under cfg["paths"]["out_dir"]):
#   R12_sweep_index.csv
#   R12_A0_vs_epsilon_rows.csv
#   R12_A0_vs_epsilon_consensus.csv
#   R12_A0_vs_epsilon_plot_<Planet>.png
#
# Behavior
#   For each epsilon in the sweep:
#     - Writes a temp config with:
#         r6.epsilon = epsilon
#         r6.filename_suffix = "eps_<value>"
#         paths.out_dir = <base_out_dir>/R12_eps_<value>
#     - Calls R6.run(conf_path=temp_cfg)
#     - Calls R8.run(conf_path=temp_cfg)
#   Because R6 supports filename suffixing and we isolate into subfolders,
#   no files are overwritten even if out_dir is shared accidentally.
#
# ASCII only. main(argv=None) and run(...) shims included.
# -----------------------------------------------------------------------------

from __future__ import annotations
import os

# Safety: pin BLAS threads unless user overrides in environment
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
from pathlib import Path
from typing import List, Tuple

import yaml
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from entanglement_validation.scripts.physics_common import (
    load_cfg, default_conf_path, ensure_out_dir
)

# Import R6 and R8 runners
try:
    from entanglement_validation.scripts.validation_r6_lrl import run as run_r6
except Exception as e:
    run_r6 = None
    print("[validation_r12] WARN: could not import run_r6:", e)

try:
    from entanglement_validation.scripts.validation_r8_peri_uncertainty import run as run_r8
except Exception as e:
    run_r8 = None
    print("[validation_r12] WARN: could not import run_r8:", e)


# ---------------------------- Helpers ----------------------------

def _write_temp_cfg(base_cfg: dict, out_dir_for_eps: Path, epsilon_val: float, L_q_val: float | None) -> Path:
    """
    Create a temp YAML config for the given epsilon (and L_q if provided):
      - paths.out_dir points to the epsilon-specific subfolder
      - r6.epsilon is set to epsilon_val
      - r6.filename_suffix is set to "eps_<value>"
      - r6.L_q is written if provided
    Returns the path to the temp yaml.
    """
    cfg = dict(base_cfg)  # shallow copy
    cfg.setdefault("paths", {})
    cfg.setdefault("r6", {})

    cfg["paths"]["out_dir"] = str(out_dir_for_eps)
    cfg["r6"]["epsilon"] = float(epsilon_val)
    cfg["r6"]["filename_suffix"] = "eps_{:.9f}".format(float(epsilon_val))
    if L_q_val is not None:
        cfg["r6"]["L_q"] = float(L_q_val)

    cfg.setdefault("r8", {})
    cfg["r8"]["r6_summary_file"] = str(out_dir_for_eps / "R6_perihelion_summary_{}.csv".format(cfg["r6"]["filename_suffix"]))

    out_dir_for_eps.mkdir(parents=True, exist_ok=True)
    tmp_path = out_dir_for_eps / "config_temp_r12.yaml"
    with open(tmp_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return tmp_path


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        print("[validation_r12] WARN could not read {}: {}".format(path, e))
        return pd.DataFrame()


def _linear_fit(x, y):
    """
    Simple line fit y = a*x + b returning (a, b).
    Centers x for numerical conditioning. Returns (nan, nan) on failure.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if len(x) < 2:
        return (np.nan, np.nan)
    xm = float(np.mean(x))
    xc = x - xm
    xs = float(np.max(np.abs(xc))) or 1.0
    X = np.column_stack([xc / xs, np.ones_like(x)])
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        a_scaled, b_scaled = float(beta[0]), float(beta[1])
        a = a_scaled / xs
        b = b_scaled - a_scaled * (xm / xs)
        return (a, b)
    except Exception:
        return (np.nan, np.nan)


def _plot_A0_vs_epsilon(planet: str, rows_cons: pd.DataFrame, out_png: Path):
    """
    Plot per-planet A0_consensus vs epsilon across the sweep. Add a simple line fit.
    """
    if rows_cons.empty:
        return
    sub = rows_cons[rows_cons["planet"] == planet].copy()
    if sub.empty:
        return
    eps = sub["epsilon"].to_numpy(dtype=float)
    A0c = sub["A0_consensus"].to_numpy(dtype=float)
    a, b = _linear_fit(eps, A0c)
    yfit = a * eps + b if np.isfinite(a) and np.isfinite(b) else None

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot(eps, A0c, marker="o", linestyle="-", label="A0_consensus")
    if yfit is not None:
        ax.plot(eps, yfit, linestyle="--", label="linear fit")
        ax.text(0.02, 0.95, "slope dA0/deps = {:.3e}".format(a),
                transform=ax.transAxes, va="top", ha="left")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("A0 consensus (rad/orbit)")
    ax.set_title("R12: {} A0 vs epsilon".format(planet))
    ax.grid(True, alpha=0.4)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("[validation_r12] wrote {}".format(out_png))


# ---------------------------- Core sweep logic ----------------------------

def _run_one_epsilon(base_cfg: dict,
                     base_out_dir: Path,
                     epsilon_val: float,
                     L_q_val: float | None,
                     reuse: bool) -> Tuple[Path, pd.DataFrame, pd.DataFrame]:
    """
    For one epsilon:
      - write temp cfg
      - run R6 compute (no_spawn) then R6 replot to force suffixed summary
      - run R8 to consume those files
      - return subfolder path and the two R8 DataFrames
    """
    subdir = base_out_dir / "R12_eps_{:.9f}".format(float(epsilon_val))
    cfg_path = _write_temp_cfg(base_cfg, subdir, epsilon_val, L_q_val)

    r8_boot = subdir / "R8_perihelion_bootstrap.csv"
    r8_cons = subdir / "R8_perihelion_consensus.csv"

    if reuse and r8_boot.exists() and r8_cons.exists():
        print("[validation_r12] reuse enabled; skipping compute for epsilon", epsilon_val)
        return subdir, pd.read_csv(r8_boot), pd.read_csv(r8_cons)

    if run_r6 is None or run_r8 is None:
        raise RuntimeError("R6 or R8 run() not available. Check imports.")

    # 1) R6 compute pass (in-process, avoids child procs on laptops)
    run_r6(conf_path=str(cfg_path), no_spawn=True)

    # 2) R6 replot pass to GUARANTEE a suffixed summary in this subfolder
    #    e.g., R6_perihelion_summary_eps_1.000000000.csv
    run_r6(conf_path=str(cfg_path), replot=True)

    # 3) R8 consumes the per-integrator CSVs (suffix-aware) and writes its outputs
    run_r8(conf_path=str(cfg_path), boot=None, check=False)

    # Safety read (won't crash if not present)
    df_boot = _safe_read_csv(r8_boot)
    df_cons = _safe_read_csv(r8_cons)
    return subdir, df_boot, df_cons



def _collect_rows(df_boot: pd.DataFrame, epsilon_val: float) -> pd.DataFrame:
    """
    Prepare per-integrator rows with epsilon column added.
    """
    if df_boot.empty:
        return pd.DataFrame()
    keep = [c for c in df_boot.columns if c in (
        "planet","integrator","p_order","rows",
        "A0_hat","A0_boot_mean","A0_boot_std","A0_boot_q16","A0_boot_q84",
        "gr_rad_per_orbit","rel_err","min_dt","max_dt"
    )]
    sub = df_boot[keep].copy() if keep else df_boot.copy()
    sub.insert(0, "epsilon", float(epsilon_val))
    return sub


def _collect_consensus(df_cons: pd.DataFrame, epsilon_val: float) -> pd.DataFrame:
    """
    Prepare per-planet consensus rows with epsilon column added.
    """
    if df_cons.empty:
        return pd.DataFrame()
    keep = [c for c in df_cons.columns if c in (
        "planet","A0_consensus","A0_consensus_std","gr_rad_per_orbit","relative_error_consensus","n_integrators"
    )]
    sub = df_cons[keep].copy() if keep else df_cons.copy()
    sub.insert(0, "epsilon", float(epsilon_val))
    return sub


# ---------------------------- CLI / Orchestrator ----------------------------

def main(argv=None):
    if argv is None:
        argv = []

    ap = argparse.ArgumentParser(
        description="validation_r12: sweep epsilon (and optional L_q) to map A0 vs parameters",
        add_help=True
    )
    ap.add_argument("--conf", default=str(default_conf_path(Path(__file__).resolve())),
                    help="Path to YAML config. Defaults to ../config.yaml relative to this script.")
    ap.add_argument("--eps", default="0.9,0.95,1.0,1.05,1.1",
                    help="Comma-separated epsilon sweep, e.g. 0.9,1.0,1.1")
    ap.add_argument("--Lq", default=None,
                    help="Optional L_q value to include in r6.L_q (placeholder).")
    ap.add_argument("--reuse", action="store_true",
                    help="If set, reuse existing R8 outputs in epsilon subfolders.")
    ap.add_argument("--check", action="store_true",
                    help="Print sanity info.")
    args = ap.parse_args(argv)

    cfg = load_cfg(Path(args.conf))
    base_out_dir = ensure_out_dir(cfg)

    # Parse epsilon list
    try:
        eps_list = [float(x.strip()) for x in str(args.eps).split(",") if x.strip() != ""]
    except Exception:
        print("[validation_r12] ERROR parsing --eps")
        return

    L_q_val = None
    if args.Lq is not None and str(args.Lq).strip() != "":
        try:
            L_q_val = float(args.Lq)
        except Exception:
            print("[validation_r12] WARN could not parse --Lq; ignoring")

    if args.check:
        print("[validation_r12] base_out_dir:", base_out_dir)
        print("[validation_r12] eps_list:", eps_list)
        print("[validation_r12] L_q:", L_q_val)

    # Sweep
    sweep_index_rows = []
    all_rows = []
    all_cons = []

    for eps in eps_list:
        subdir, df_boot, df_cons = _run_one_epsilon(cfg, base_out_dir, eps, L_q_val, reuse=args.reuse)
        sweep_index_rows.append(dict(epsilon=float(eps), subfolder=str(subdir)))

        rows_pi = _collect_rows(df_boot, eps)
        rows_pp = _collect_consensus(df_cons, eps)

        if not rows_pi.empty:
            all_rows.append(rows_pi)
        if not rows_pp.empty:
            all_cons.append(rows_pp)

    # Write sweep index
    if sweep_index_rows:
        idx_df = pd.DataFrame(sweep_index_rows)
        idx_path = base_out_dir / "R12_sweep_index.csv"
        idx_df.to_csv(idx_path, index=False)
        print("[validation_r12] wrote {}".format(idx_path))

    # Write per-integrator rows
    rows_out = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    rows_path = base_out_dir / "R12_A0_vs_epsilon_rows.csv"
    rows_out.to_csv(rows_path, index=False)
    print("[validation_r12] wrote {}".format(rows_path))

    # Write per-planet consensus
    cons_out = pd.concat(all_cons, ignore_index=True) if all_cons else pd.DataFrame()
    cons_path = base_out_dir / "R12_A0_vs_epsilon_consensus.csv"
    cons_out.to_csv(cons_path, index=False)
    print("[validation_r12] wrote {}".format(cons_path))

    # Plots per planet
    if not cons_out.empty and "planet" in cons_out.columns:
        for planet in sorted(cons_out["planet"].dropna().unique()):
            out_png = base_out_dir / "R12_A0_vs_epsilon_plot_{}.png".format(planet)
            _plot_A0_vs_epsilon(planet, cons_out, out_png)

    print("[validation_r12] done.")


def run(conf_path=None, eps=None, L_q=None, reuse=False, check=False):
    """
    Shim for orchestrator and IDE use.
    """
    argv = []
    if conf_path:
        argv += ["--conf", str(conf_path)]
    if eps is not None:
        if isinstance(eps, (list, tuple)):
            argv += ["--eps", ",".join(str(float(x)) for x in eps)]
        else:
            argv += ["--eps", str(eps)]
    if L_q is not None:
        argv += ["--Lq", str(float(L_q))]
    if reuse:
        argv += ["--reuse"]
    if check:
        argv += ["--check"]
    return main(argv)


if __name__ == "__main__":
    main([])
