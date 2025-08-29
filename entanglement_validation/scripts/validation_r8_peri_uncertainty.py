#!/usr/bin/env python3

# validation_r8_peri_uncertainty.py
# -----------------------------------------------------------------------------
# Purpose
#   Estimate uncertainty on A0 (dt -> 0 extrapolated perihelion advance) for
#   each planet x integrator using a numerically stable linear fit and a
#   residual bootstrap. Also produce a per-planet consensus across integrators.
#
# Inputs (produced by R6)
#   R6_perihelion_convergence_{Planet}_{Integrator}[<suffix>].csv
#     columns typically include:
#       dt, advance_rad_per_orbit, gr_rad_per_orbit, epsilon_used, periods
#
# Config (same YAML used by R6)
#   paths.out_dir
#   r6.planets
#   r6.integrators
#   r6.filename_suffix   # optional, e.g. "eps_1.000000000"
#
# Outputs
#   R8_perihelion_bootstrap.csv     # per planet x integrator uncertainty
#   R8_perihelion_consensus.csv     # per planet consensus across integrators
#
# Notes
#   - ASCII-only.
#   - Works from an IDE without CLI args. Defaults to ../config.yaml relative
#     to this file. You can override with --conf C:\path\to\config.yaml
#   - Uses a stable linear fit for y vs x where x = dt**p_order, y = advance.
#   - Uses a residual bootstrap to avoid singular resamples.
# -----------------------------------------------------------------------------

from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# =============================================================================
# Config helpers
# =============================================================================

def _find_r6_summary(cfg: dict, cli_path: str = "") -> str:
    """
    Return a path to the R6 summary CSV, trying in this order:
      1) CLI override (cli_path) if provided
      2) cfg['r8']['r6_summary_file'] if provided
      3) suffixed file in cfg['paths']['out_dir'] using cfg['r6']['filename_suffix']
      4) unsuffixed file in cfg['paths']['out_dir']
    Returns "" if nothing found.
    """
    if cli_path and os.path.isfile(cli_path):
        return cli_path

    out_dir = cfg.get("paths", {}).get("out_dir", "")
    sfx = cfg.get("r6", {}).get("filename_suffix", "")
    hint = cfg.get("r8", {}).get("r6_summary_file", "")

    if hint and os.path.isfile(hint):
        return hint

    if out_dir and sfx:
        p = os.path.join(out_dir, "R6_perihelion_summary_{}.csv".format(sfx))
        if os.path.isfile(p):
            return p

    if out_dir:
        p = os.path.join(out_dir, "R6_perihelion_summary.csv")
        if os.path.isfile(p):
            return p

    return ""


def load_cfg(conf_path: Path) -> dict:
    with open(conf_path, "r") as f:
        return yaml.safe_load(f)

def default_conf_path(start_file: Path) -> Path:
    # Default to ../config.yaml relative to this script
    return (start_file.parent.parent / "config.yaml").resolve()

def ensure_out_dir(cfg: dict) -> Path:
    out_dir = Path(cfg["paths"]["out_dir"]).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def _suffix_from_cfg(cfg: dict) -> str:
    """
    Return "" or a filename suffix like "_eps_1.000000000" derived from
    cfg['r6']['filename_suffix'] if present.
    """
    s = str(cfg.get("r6", {}).get("filename_suffix", "")).strip()
    return ("_" + s) if s else ""


# =============================================================================
# File discovery
# =============================================================================

def discover_planets_from_files(out_dir: Path, allowed: set[str] | None) -> list[str]:
    """
    Find planets that actually have R6 CSVs on disk.
    Works for both suffixed and unsuffixed filenames:
      R6_perihelion_convergence_{Planet}_{Integrator}.csv
      R6_perihelion_convergence_{Planet}_{Integrator}_<suffix>.csv
    """
    found = set()
    for p in out_dir.glob("R6_perihelion_convergence_*_*.csv"):
        name = p.name.replace(".csv", "")
        parts = name.split("_")
        # Expected start: ["R6","perihelion","convergence","{Planet}","{Integrator}", ...maybe suffix parts...]
        if len(parts) >= 5 and parts[0] == "R6" and parts[1] == "perihelion" and parts[2] == "convergence":
            planet = parts[3]
            if (allowed is None) or (planet in allowed):
                found.add(planet)
    return sorted(found)


def load_convergence(out_dir: Path, planet: str, integrator: str, suffix: str = "") -> pd.DataFrame | None:
    """
    Load per-integrator convergence CSV. If suffix is provided (e.g. '_eps_1.000000000'),
    look for the suffixed filename first, then fall back to unsuffixed.
    """
    candidates = []
    if suffix:
        candidates.append(out_dir / f"R6_perihelion_convergence_{planet}_{integrator}{suffix}.csv")
    candidates.append(out_dir / f"R6_perihelion_convergence_{planet}_{integrator}.csv")

    for path in candidates:
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception as e:
                print(f"[validation_r8] ERROR reading {path.name}: {e}")
                return None

    print("[validation_r8] WARN missing per-integrator CSV for {} {} (tried: {})"
          .format(planet, integrator, ", ".join(p.name for p in candidates)))
    return None


# =============================================================================
# Stable linear fit and residual bootstrap
# =============================================================================

def _stable_fit_intercept(x_raw, y_raw):
    """
    Fit y = a*x + b using a numerically stable approach and return b at x=0.
    """
    x = np.asarray(x_raw, dtype=float).ravel()
    y = np.asarray(y_raw, dtype=float).ravel()

    if x.size == 0:
        return np.nan
    if x.size == 1:
        return float(y[0])

    # Center and scale x for numerical stability
    x_mean = float(np.mean(x))
    x_centered = x - x_mean
    x_scale = float(np.max(np.abs(x_centered))) or 1.0
    x_scaled = x_centered / x_scale

    X = np.column_stack([x_scaled, np.ones_like(x_scaled)])

    sol, residuals, rank, svals = np.linalg.lstsq(X, y, rcond=None)
    cond = (svals[0] / svals[-1]) if (svals.size >= 2 and svals[-1] != 0) else np.inf
    if rank < 2 or not np.isfinite(cond) or cond > 1e12:
        lam = 1e-12
        XtX = X.T @ X
        XtY = X.T @ y
        beta = np.linalg.solve(XtX + lam * np.eye(2), XtY)
        a_hat, b_scaled = float(beta[0]), float(beta[1])
    else:
        a_hat, b_scaled = float(sol[0]), float(sol[1])

    # Intercept at x=0 in original units
    b_at_zero = b_scaled - a_hat * (x_mean / x_scale)
    return float(b_at_zero)


def fit_A0_intercept(dts, advs, p_order):
    """
    Compute A0 as intercept of y vs x with x = dt**p_order.
    Uses stable solver. If fewer than 2 unique x, return mean(y).
    """
    x = np.asarray(dts, dtype=float) ** float(p_order)
    y = np.asarray(advs, dtype=float)
    if len(np.unique(np.round(x, 12))) < 2:
        return float(np.mean(y)) if y.size else np.nan
    return _stable_fit_intercept(x, y)


def _stable_fit_full(x_raw, y_raw):
    """
    Fit y = a*x + b and return slope (original units), intercept at x=0, and yhat.
    """
    x = np.asarray(x_raw, dtype=float).ravel()
    y = np.asarray(y_raw, dtype=float).ravel()

    x_mean = float(np.mean(x))
    x_centered = x - x_mean
    x_scale = float(np.max(np.abs(x_centered))) or 1.0
    x_scaled = x_centered / x_scale

    X = np.column_stack([x_scaled, np.ones_like(x_scaled)])
    sol, residuals, rank, svals = np.linalg.lstsq(X, y, rcond=None)
    cond = (svals[0] / svals[-1]) if (svals.size >= 2 and svals[-1] != 0) else np.inf
    if rank < 2 or not np.isfinite(cond) or cond > 1e12:
        lam = 1e-12
        XtX = X.T @ X
        XtY = X.T @ y
        beta = np.linalg.solve(XtX + lam * np.eye(2), XtY)
    else:
        beta = sol

    a_hat, b_scaled = float(beta[0]), float(beta[1])
    yhat = a_hat * ((x - x_mean) / x_scale) + b_scaled
    b_at_zero = b_scaled - a_hat * (x_mean / x_scale)
    slope_orig = a_hat / x_scale
    return slope_orig, float(b_at_zero), yhat


def bootstrap_A0_residual(dts, advs, p_order, n_boot=500, rng=None):
    """
    Residual bootstrap for intercept A0.
    """
    if rng is None:
        rng = np.random.default_rng(12345)

    x = np.asarray(dts, dtype=float) ** float(p_order)
    y = np.asarray(advs, dtype=float)

    n = y.size
    if n == 0:
        return dict(mean=np.nan, std=np.nan, q16=np.nan, q84=np.nan)

    if len(np.unique(np.round(x, 12))) < 2:
        a0 = fit_A0_intercept(dts, advs, p_order)
        return dict(mean=a0, std=np.nan, q16=a0, q84=a0)

    slope, intercept0, yhat = _stable_fit_full(x, y)
    resid = y - yhat

    stats = []
    idx = np.arange(n)
    for _ in range(int(n_boot)):
        e_star = resid[rng.choice(idx, size=n, replace=True)]
        y_star = yhat + e_star
        a0_star = _stable_fit_intercept(x, y_star)
        stats.append(a0_star)

    stats = np.array(stats, dtype=float)
    return dict(
        mean=float(np.nanmean(stats)),
        std=float(np.nanstd(stats, ddof=1)),
        q16=float(np.nanpercentile(stats, 16)),
        q84=float(np.nanpercentile(stats, 84)),
    )


# =============================================================================
# Per-integrator summarization and consensus
# =============================================================================

def summarize_planet_integrator(df: pd.DataFrame, planet: str, integrator: str, p_order: int, n_boot: int):
    """
    Summarize one planet x integrator.
    """
    if df is None or df.empty:
        return None

    required = ["dt", "advance_rad_per_orbit"]
    for col in required:
        if col not in df.columns:
            print("[validation_r8] WARN {} {} missing column: {}".format(planet, integrator, col))
            return None

    dts = df["dt"].to_numpy(dtype=float)
    adv = df["advance_rad_per_orbit"].to_numpy(dtype=float)

    a0_hat = fit_A0_intercept(dts, adv, p_order)
    boot = bootstrap_A0_residual(dts, adv, p_order, n_boot=n_boot)

    gr = float(df["gr_rad_per_orbit"].iloc[0]) if "gr_rad_per_orbit" in df.columns else np.nan
    eps = float(df["epsilon_used"].iloc[0]) if "epsilon_used" in df.columns else np.nan
    periods = float(df["periods"].iloc[0]) if "periods" in df.columns else np.nan
    rows = int(len(df))

    rel_err = abs(a0_hat - gr) / gr if (np.isfinite(a0_hat) and np.isfinite(gr) and gr != 0.0) else np.nan

    return dict(
        planet=planet,
        integrator=integrator,
        p_order=int(p_order),
        rows=rows,
        A0_hat=a0_hat,
        A0_boot_mean=boot["mean"],
        A0_boot_std=boot["std"],
        A0_boot_q16=boot["q16"],
        A0_boot_q84=boot["q84"],
        gr_rad_per_orbit=gr,
        rel_err=rel_err,
        epsilon_used=eps,
        periods=periods,
        min_dt=float(np.min(dts) if len(dts) else np.nan),
        max_dt=float(np.max(dts) if len(dts) else np.nan),
    )


def consensus_per_planet(rows_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine integrators per planet using inverse-variance weights on A0_boot_std.
    If std is missing or zero, fall back to equal weights.
    """
    if rows_df is None or rows_df.empty:
        return pd.DataFrame()

    out = []
    for planet, sub in rows_df.groupby("planet"):
        vals = sub["A0_boot_mean"].to_numpy(dtype=float)
        sigs = sub["A0_boot_std"].to_numpy(dtype=float)
        grs = sub["gr_rad_per_orbit"].to_numpy(dtype=float)

        if np.any(~np.isfinite(sigs)) or np.any(sigs <= 0):
            w = np.ones_like(vals)
        else:
            w = 1.0 / np.clip(sigs, 1e-30, np.inf) ** 2

        if len(vals) == 0:
            continue

        a0_cons = float(np.sum(w * vals) / np.sum(w))
        a0_cons_std = float((1.0 / np.sqrt(np.sum(w))) if np.all(np.isfinite(w)) else np.nan)

        gr = float(np.nanmean(grs))
        rel_err_cons = abs(a0_cons - gr) / gr if (np.isfinite(a0_cons) and np.isfinite(gr) and gr != 0.0) else np.nan

        out.append(dict(
            planet=planet,
            A0_consensus=a0_cons,
            A0_consensus_std=a0_cons_std,
            gr_rad_per_orbit=gr,
            relative_error_consensus=rel_err_cons,
            n_integrators=int(len(sub)),
        ))

    return pd.DataFrame(out)


# =============================================================================
# Main
# =============================================================================

def main(argv=None):
    if argv is None:
        argv = []

    ap = argparse.ArgumentParser(description="validation_r8: bootstrap uncertainty and consensus")
    ap.add_argument("--conf",
                    default=str(default_conf_path(Path(__file__).resolve())),
                    help="Path to YAML config. Defaults to ../config.yaml relative to this script.")
    ap.add_argument("--boot", type=int, default=500,
                    help="Bootstrap iterations per integrator. Default 500.")
    ap.add_argument("--check", action="store_true",
                    help="Print sanity info.")
    ap.add_argument("--r6_summary", default=None,
                    help="Optional path to an R6 summary CSV to use.")

    args = ap.parse_args(argv)

    cfg = load_cfg(Path(args.conf))
    out_dir = ensure_out_dir(cfg)
    sfx = _suffix_from_cfg(cfg)

    # Scope from config
    cfg_planets = list(cfg.get("r6", {}).get("planets", ["Mercury", "Venus", "Earth"]))
    cfg_integrators = list(cfg.get("r6", {}).get("integrators", ["vv", "rk4"]))
    allowed = set(cfg_planets)

    # Discover which planets actually exist on disk
    planets = discover_planets_from_files(out_dir, allowed) or cfg_planets

    # Integrator -> polynomial order mapping
    porder_map = {"vv": 2, "rk4": 4, "yoshida4": 4}

    if args.check:
        print("[validation_r8] out_dir:", out_dir)
        print("[validation_r8] suffix:", sfx)
        print("[validation_r8] planets:", planets)
        print("[validation_r8] integrators:", cfg_integrators)

    rows = []
    for planet in planets:
        for integ in cfg_integrators:
            df = load_convergence(out_dir, planet, integ, suffix=sfx)
            if df is None or df.empty:
                continue
            p_order = porder_map.get(integ, 2)
            row = summarize_planet_integrator(df, planet, integ, p_order, n_boot=args.boot)
            if row:
                rows.append(row)

    if not rows:
        print("[validation_r8] nothing to summarize.")
        return

    r8_rows = pd.DataFrame(rows).sort_values(["planet", "integrator"]).reset_index(drop=True)
    r8_rows_path = out_dir / "R8_perihelion_bootstrap.csv"
    r8_rows.to_csv(r8_rows_path, index=False)
    print("[validation_r8] wrote {}".format(r8_rows_path))

    # Per-planet consensus across integrators
    r8_cons = consensus_per_planet(r8_rows)
    r8_cons_path = out_dir / "R8_perihelion_consensus.csv"
    r8_cons.to_csv(r8_cons_path, index=False)
    print("[validation_r8] wrote {}".format(r8_cons_path))


def run(conf_path=None, boot=None, check=False):
    """
    Thin adapter so run_validations.py can import and call run().
    """
    argv = []
    if conf_path:
        argv += ["--conf", str(conf_path)]
    if boot is not None:
        argv += ["--boot", str(int(boot))]
    if check:
        argv += ["--check"]
    return main(argv)


if __name__ == "__main__":
    main([])
