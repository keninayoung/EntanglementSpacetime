#!/usr/bin/env python3

# validation_r11_edg_constraints.py
# -----------------------------------------------------------------------------
# Purpose
#   Infer constraints on the EDG coupling "epsilon" using perihelion results.
#   We assume, in the weak-field regime of R6, that the relativistic advance
#   scales approximately linearly with epsilon. Thus:
#       A0_pred(epsilon) ~ epsilon * GR_target
#   Given per-integrator bootstrap means from R8 (A0_boot_mean) and the same
#   GR target, a simple estimator is:
#       epsilon_hat = A0_boot_mean / GR_target
#       sigma_epsilon = A0_boot_std / GR_target
#
# Inputs (under cfg["paths"]["out_dir"])
#   R8_perihelion_bootstrap.csv   # from validation_r8_peri_uncertainty.py
#
# Outputs (under cfg["paths"]["out_dir"])
#   R11_edg_constraints_rows.csv      # per planet x integrator epsilon estimates
#   R11_edg_constraints_consensus.csv # per planet consensus epsilon, plus global
#   R11_edg_constraints.md            # human-readable table (Markdown)
#   R11_edg_constraints.tex           # LaTeX table
#
# Notes
#   - ASCII only.
#   - IDE friendly. No required CLI args; has a run() shim for orchestrator.
#   - This step does NOT re-run integrators. It uses R8 outputs.
#   - For integrators with zero bootstrap std, we apply a tiny floor to avoid
#     infinite weights in the consensus. Adjust via --std_floor if needed.
# -----------------------------------------------------------------------------

from __future__ import annotations
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import yaml

# Import config helpers from your package
from entanglement_validation.scripts.physics_common import (
    load_cfg, default_conf_path, ensure_out_dir
)

# =============================================================================
# IO helpers
# =============================================================================

def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print("[validation_r11] WARN missing {}".format(path.name))
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        print("[validation_r11] ERROR reading {}: {}".format(path.name, e))
        return pd.DataFrame()

def _fmt_float(x, sig=6):
    try:
        f = float(x)
    except Exception:
        return str(x)
    if f == 0.0:
        return "0"
    if abs(f) < 1e-3 or abs(f) >= 1e4:
        return "{:.{}e}".format(f, sig)
    return "{:.{}g}".format(f, sig)

def _md_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_No data._\n"
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        vals = [str(row.get(c, "")) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"

def _latex_table(df: pd.DataFrame, caption="EDG constraints", label="tab:r11") -> str:
    if df is None or df.empty:
        return "% No data.\n"
    cols = list(df.columns)
    colspec = " | ".join(["l"] * len(cols))
    lines = []
    lines.append("\\begin{table}[h!]")
    lines.append("\\centering")
    lines.append("\\caption{%s}" % caption)
    lines.append("\\label{%s}" % label)
    lines.append("\\begin{tabular}{%s}" % colspec)
    lines.append("\\hline")
    lines.append(" & ".join(cols) + " \\\\")
    lines.append("\\hline")
    for _, row in df.iterrows():
        vals = [str(row.get(c, "")) for c in cols]
        lines.append(" & ".join(vals) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"

# =============================================================================
# Core math
# =============================================================================

def per_integrator_eps_rows(r8_rows: pd.DataFrame, std_floor: float) -> pd.DataFrame:
    """
    Build per-(planet, integrator) epsilon estimates from R8 per-integrator rows.
    Requires columns: planet, integrator, A0_boot_mean, A0_boot_std, gr_rad_per_orbit
    """
    needed = ["planet","integrator","A0_boot_mean","A0_boot_std","gr_rad_per_orbit"]
    for col in needed:
        if col not in r8_rows.columns:
            raise ValueError("R8 rows missing required column: {}".format(col))

    out = []
    for _, r in r8_rows.iterrows():
        planet = r["planet"]
        integ = r["integrator"]
        gr = float(r["gr_rad_per_orbit"]) if pd.notna(r["gr_rad_per_orbit"]) else np.nan
        a0m = float(r["A0_boot_mean"]) if pd.notna(r["A0_boot_mean"]) else np.nan
        a0s = float(r["A0_boot_std"]) if pd.notna(r["A0_boot_std"]) else np.nan

        if not np.isfinite(gr) or gr == 0.0 or not np.isfinite(a0m):
            eps_hat = np.nan
            eps_std = np.nan
        else:
            eps_hat = a0m / gr
            # propagate uncertainty; apply std floor to avoid infinite weights
            if not np.isfinite(a0s):
                a0s = np.nan
            if not np.isfinite(a0s) or a0s <= 0.0:
                eps_std = std_floor
            else:
                eps_std = max(a0s / abs(gr), std_floor)

        out.append(dict(
            planet=planet,
            integrator=integ,
            epsilon_hat=eps_hat,
            epsilon_std=eps_std,
            A0_boot_mean=a0m,
            A0_boot_std=a0s,
            gr_rad_per_orbit=gr
        ))

    df = pd.DataFrame(out)
    # Nice formatting pass for saving human-readable CSVs
    for col in ["epsilon_hat","epsilon_std","A0_boot_mean","A0_boot_std","gr_rad_per_orbit"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df

def inv_var_consensus(values: np.ndarray, sigmas: np.ndarray) -> Tuple[float,float,int]:
    """
    Inverse-variance weighted mean and its std. Returns (mean, std, n_used).
    Ignores any entries with non-finite sigma or sigma <= 0.
    """
    vals = np.asarray(values, dtype=float)
    s = np.asarray(sigmas, dtype=float)
    mask = np.isfinite(vals) & np.isfinite(s) & (s > 0)
    if not np.any(mask):
        return (np.nan, np.nan, 0)
    w = 1.0 / np.clip(s[mask], 1e-30, np.inf)**2
    mean = float(np.sum(w * vals[mask]) / np.sum(w))
    std = float(1.0 / np.sqrt(np.sum(w)))
    n_used = int(np.sum(mask))
    return (mean, std, n_used)

def per_planet_consensus(rows: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-planet epsilon consensus from per-integrator rows.
    """
    out = []
    for planet, sub in rows.groupby("planet"):
        eps = sub["epsilon_hat"].to_numpy(dtype=float)
        sig = sub["epsilon_std"].to_numpy(dtype=float)
        mean, std, n_used = inv_var_consensus(eps, sig)
        out.append(dict(
            planet=planet,
            epsilon_consensus=mean,
            epsilon_consensus_std=std,
            n_integrators=n_used
        ))
    return pd.DataFrame(out)

def global_consensus(rows_cons: pd.DataFrame) -> Tuple[float,float,int]:
    """
    Global epsilon consensus across planets, using per-planet consensus and std.
    """
    eps = rows_cons["epsilon_consensus"].to_numpy(dtype=float)
    sig = rows_cons["epsilon_consensus_std"].to_numpy(dtype=float)
    return inv_var_consensus(eps, sig)

def ci_68_95(mean: float, std: float) -> Tuple[str, str]:
    """
    Return compact strings for 68% and 95% intervals assuming Gaussian.
    """
    if not np.isfinite(mean) or not np.isfinite(std):
        return ("NA", "NA")
    lo68, hi68 = mean - std, mean + std
    lo95, hi95 = mean - 1.96*std, mean + 1.96*std
    s68 = "[{:.6g}, {:.6g}]".format(lo68, hi68)
    s95 = "[{:.6g}, {:.6g}]".format(lo95, hi95)
    return (s68, s95)

# =============================================================================
# Report writers
# =============================================================================

def write_md(out_dir: Path, rows_pi: pd.DataFrame, rows_pp: pd.DataFrame, g_mean: float, g_std: float):
    md_path = out_dir / "R11_edg_constraints.md"
    lines = []
    lines.append("# R11: EDG Epsilon Constraints")
    lines.append("")
    lines.append("This report infers epsilon from R8 perihelion bootstrap means using epsilon_hat = A0_boot_mean / GR.")
    lines.append("")
    # Per-integrator
    if not rows_pi.empty:
        tmp = rows_pi.copy()
        for col in ["epsilon_hat","epsilon_std","A0_boot_mean","A0_boot_std","gr_rad_per_orbit"]:
            if col in tmp.columns:
                tmp[col] = tmp[col].map(_fmt_float)
        lines.append("## Per Integrator")
        lines.append(_md_table(tmp))
    else:
        lines.append("_No per-integrator data._\n")
    # Per-planet consensus
    if not rows_pp.empty:
        tmp = rows_pp.copy()
        for col in ["epsilon_consensus","epsilon_consensus_std"]:
            if col in tmp.columns:
                tmp[col] = tmp[col].map(_fmt_float)
        lines.append("## Per Planet Consensus")
        lines.append(_md_table(tmp))
    else:
        lines.append("_No per-planet consensus data._\n")
    # Global
    if np.isfinite(g_mean) and np.isfinite(g_std):
        s68, s95 = ci_68_95(g_mean, g_std)
        lines.append("## Global Consensus")
        lines.append("")
        lines.append("- epsilon_global = {} +/- {} (68%)".format(_fmt_float(g_mean), _fmt_float(g_std)))
        lines.append("- 68% CI: {}".format(s68))
        lines.append("- 95% CI: {}".format(s95))
        lines.append("")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("[validation_r11] wrote {}".format(md_path))

def write_tex(out_dir: Path, rows_pi: pd.DataFrame, rows_pp: pd.DataFrame, g_mean: float, g_std: float):
    tex_path = out_dir / "R11_edg_constraints.tex"
    lines = []
    lines.append("\\section{R11: EDG Epsilon Constraints}")
    lines.append("We infer $\\epsilon$ from R8 perihelion bootstrap means using $\\hat{\\epsilon} = A0\\_\\text{boot\\_mean} / \\text{GR}$.")
    lines.append("")
    if not rows_pi.empty:
        tmp = rows_pi.copy()
        for col in ["epsilon_hat","epsilon_std","A0_boot_mean","A0_boot_std","gr_rad_per_orbit"]:
            if col in tmp.columns:
                tmp[col] = tmp[col].map(_fmt_float)
        lines.append(_latex_table(tmp, caption="Per-integrator epsilon estimates", label="tab:r11_per_integrator"))
    if not rows_pp.empty:
        tmp = rows_pp.copy()
        for col in ["epsilon_consensus","epsilon_consensus_std"]:
            if col in tmp.columns:
                tmp[col] = tmp[col].map(_fmt_float)
        lines.append(_latex_table(tmp, caption="Per-planet epsilon consensus", label="tab:r11_per_planet"))
    if np.isfinite(g_mean) and np.isfinite(g_std):
        s68, s95 = ci_68_95(g_mean, g_std)
        lines.append("\\subsection*{Global Consensus}")
        lines.append("Global $\\epsilon$: {} $\\pm$ {} (68\\%). 68\\% CI: {}. 95\\% CI: {}."
                     .format(_fmt_float(g_mean), _fmt_float(g_std), s68, s95))
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("[validation_r11] wrote {}".format(tex_path))

# =============================================================================
# Main
# =============================================================================

def main(argv=None):
    if argv is None:
        argv = []

    ap = argparse.ArgumentParser(description="validation_r11: EDG epsilon constraints from R8")
    ap.add_argument("--conf", default=str(default_conf_path(Path(__file__).resolve())),
                    help="Path to YAML config. Defaults to ../config.yaml relative to this script.")
    ap.add_argument("--std_floor", type=float, default=1e-12,
                    help="Floor on epsilon std to avoid infinite weights. Default 1e-12.")
    ap.add_argument("--check", action="store_true",
                    help="Print sanity info.")
    args = ap.parse_args(argv)

    cfg = load_cfg(Path(args.conf))
    out_dir = ensure_out_dir(cfg)

    # Load R8 per-integrator results
    r8_rows = _read_csv(out_dir / "R8_perihelion_bootstrap.csv")
    if args.check:
        print("[validation_r11] r8_rows shape:", getattr(r8_rows, "shape", None))

    if r8_rows.empty:
        print("[validation_r11] nothing to process (missing R8_perihelion_bootstrap.csv).")
        return

    # Build per-integrator epsilon rows
    rows_pi = per_integrator_eps_rows(r8_rows, std_floor=float(args.std_floor))
    rows_pi_path = out_dir / "R11_edg_constraints_rows.csv"
    rows_pi.to_csv(rows_pi_path, index=False)
    print("[validation_r11] wrote {}".format(rows_pi_path))

    # Per-planet consensus
    rows_pp = per_planet_consensus(rows_pi)
    rows_pp_path = out_dir / "R11_edg_constraints_consensus.csv"
    rows_pp.to_csv(rows_pp_path, index=False)
    print("[validation_r11] wrote {}".format(rows_pp_path))

    # Global consensus
    g_mean, g_std, g_n = global_consensus(rows_pp)
    if args.check:
        print("[validation_r11] global epsilon:", g_mean, "+/-", g_std, "(n_planets used =", g_n, ")")

    # Human-readable snippets
    write_md(out_dir, rows_pi, rows_pp, g_mean, g_std)
    write_tex(out_dir, rows_pi, rows_pp, g_mean, g_std)


def run(conf_path=None, std_floor=None, check=False):
    """
    Shim so run_validations.py can import and call run().

    Parameters
    ----------
    conf_path : str or Path or None
        Optional path to config.yaml. If None, script default is used.
    std_floor : float or None
        Floor on epsilon std to avoid infinite weights. If None, script default.
    check : bool
        If True, print sanity info.
    """
    argv = []
    if conf_path:
        argv += ["--conf", str(conf_path)]
    if std_floor is not None:
        argv += ["--std_floor", str(float(std_floor))]
    if check:
        argv += ["--check"]
    return main(argv)


if __name__ == "__main__":
    main([])
