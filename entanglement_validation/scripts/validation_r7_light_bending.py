#!/usr/bin/env python3

# =============================================================================
# R7: Post-processing and consolidation for R6 perihelion benchmark
# -----------------------------------------------------------------------------
# WHAT THIS DOES
#   - Loads config.yaml and uses cfg["paths"]["out_dir"] (same as R6).
#   - Loads the single R6 summary file:  R6_perihelion_summary.csv
#   - Scans R6 per-run CSVs:             R6_perihelion_convergence_{Planet}_{Integrator}.csv
#   - Normalizes any "list-like" cells to scalars, then to numeric (avoids int(list) errors).
#   - Aggregates per planet x integrator metrics, merges with R6 summary fields.
#   - Writes R7_summary.csv in out_dir.
#
# DESIGN GOALS
#   - No path confusion: everything runs under cfg["paths"]["out_dir"].
#   - Planets come from cfg["r6"]["planets"], but we also auto-discover from
#     available filenames to avoid processing planets not produced by R6.
#   - Integrators default to ["vv","rk4"] but respect cfg["r6"]["integrators"].
#   - Defensive code: missing files and empty dataframes are logged and skipped.
#   - ASCII-only file. No Unicode characters.
#
# OUTPUT
#   out_dir / R7_summary.csv
#
# USAGE
#   python R7.py --conf /path/to/config.yaml
#   python R7.py --conf /path/to/config.yaml --check   # print quick sanity info
#
# =============================================================================

import os
import sys
import re
import argparse
from pathlib import Path
import ast
import yaml
import numpy as np
import pandas as pd


# =============================================================================
# Utility Helpers
# =============================================================================
def _default_conf_path() -> Path:
    # Same idea as R6: ../config.yaml relative to this file
    here = Path(__file__).resolve()
    return (here.parent.parent / "config.yaml").resolve()

def load_cfg(conf_path: Path) -> dict:
    """
    Purpose:
      Load YAML configuration exactly as R6 does (same keys).

    Returns:
      dict cfg
    """
    with open(conf_path, "r") as f:
        return yaml.safe_load(f)


def find_r6_summary(out_dir: Path) -> Path:
    """
    Purpose:
      Resolve the single R6 summary file. R6 writes:
        R6_perihelion_summary.csv

    Returns or raises:
      Path to the file or FileNotFoundError if absent.
    """
    candidates = [
        out_dir / "R6_perihelion_summary.csv",
        out_dir / "R6_perihelion_summary",
    ]
    # Be tolerant of other extensions if present
    candidates.extend(out_dir.glob("R6_perihelion_summary.*"))
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    raise FileNotFoundError(
        "[R7] Could not find R6 summary in {} (looked for R6_perihelion_summary[.csv|.*]).".format(out_dir)
    )


def discover_planets(out_dir: Path, allowed=None) -> list:
    """
    Purpose:
      Harvest planet names from filenames that R6 actually produced:
        R6_perihelion_convergence_{Planet}_{Integrator}.csv

    Returns:
      Sorted unique list of planets, optionally filtered by 'allowed' set.
    """
    pat = re.compile(r"^R6_perihelion_convergence_([A-Za-z]+)_(.+)\.csv$")
    found = set()
    for p in out_dir.glob("R6_perihelion_convergence_*_*.csv"):
        m = pat.match(p.name)
        if m:
            found.add(m.group(1))
    planets = sorted(found)
    if allowed:
        planets = [p for p in planets if p in allowed]
    return planets


def convergence_path(out_dir: Path, planet: str, integrator: str) -> Path:
    """
    Purpose:
      Construct expected path to R6 per-run CSV for (planet, integrator).
    """
    return out_dir / "R6_perihelion_convergence_{}_{}.csv".format(planet, integrator)


# -----------------------------------------------------------------------------
# "Listy" to scalar normalizer + numeric coercion
# -----------------------------------------------------------------------------

def _coerce_listy_to_scalar(x):
    """
    Purpose:
      - If x is a Python list, return its first element (or None if empty).
      - If x is a string that looks like a list (e.g., "[1234]"), parse safely
        with ast.literal_eval and return the first element if it is a list.
      - Otherwise return x unchanged.

      This prevents int() or numeric casts from failing on list types.
    """
    if isinstance(x, list):
        return x[0] if x else None

    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list) and parsed:
                    return parsed[0]
            except Exception:
                # Fallback: simple bracket strip and first token
                try:
                    return s.strip("[]").split(",")[0].strip()
                except Exception:
                    pass

    return x


def to_numeric_series(series: pd.Series) -> pd.Series:
    """
    Purpose:
      Apply list-to-scalar normalization and then coerce to numeric.
      Handles "1,000" -> "1000" and "1_000" -> "1000".
    """
    s = series.map(_coerce_listy_to_scalar)
    s = s.astype(str).str.replace(",", "").str.replace("_", "", regex=False)
    return pd.to_numeric(s, errors="coerce")


# -----------------------------------------------------------------------------
# Per-DF summarization
# -----------------------------------------------------------------------------

_PORDER_MAP = {
    "vv": 2,
    "rk4": 4,
    "yoshida4": 4,
}

def summarize_convergence_df(df: pd.DataFrame, planet: str, integrator: str):
    """
    Purpose:
      Produce a compact set of rollups for one planet x integrator.

    Returns:
      dict with metrics or None if df is empty or invalid.
    """
    if df is None or df.empty:
        print("[R7] Skip {} {}: empty dataframe".format(planet, integrator))
        return None

    # Normalize expected numeric columns (only if present)
    for col in [
        "steps_per_orbit",
        "dt",
        "advance_rad_per_orbit",
        "advance_lrl_rad_per_orbit",
        "gr_rad_per_orbit",
        "epsilon_used",
        "periods",
    ]:
        if col in df.columns:
            df[col] = to_numeric_series(df[col])

    rows = int(len(df))
    dt_min = float(df["dt"].min()) if "dt" in df.columns and rows else float("nan")
    dt_max = float(df["dt"].max()) if "dt" in df.columns and rows else float("nan")
    adv_mean = float(df["advance_rad_per_orbit"].mean()) if "advance_rad_per_orbit" in df.columns and rows else float("nan")
    adv_std = float(df["advance_rad_per_orbit"].std()) if "advance_rad_per_orbit" in df.columns and rows else float("nan")
    gr_val = float(df["gr_rad_per_orbit"].iloc[0]) if "gr_rad_per_orbit" in df.columns and rows else float("nan")
    eps_val = float(df["epsilon_used"].iloc[0]) if "epsilon_used" in df.columns and rows else float("nan")
    periods = int(df["periods"].iloc[0]) if "periods" in df.columns and rows and pd.notna(df["periods"].iloc[0]) else None

    # Optional: recompute A0 (sanity) using the same logic order as R6
    order = _PORDER_MAP.get(integrator, 2)
    A0 = float("nan")
    rel_err = float("nan")
    if "dt" in df.columns and "advance_rad_per_orbit" in df.columns and rows:
        dts_arr = df["dt"].to_numpy(dtype=float)
        adv_arr = df["advance_rad_per_orbit"].to_numpy(dtype=float)
        if len(dts_arr) >= 2:
            x = dts_arr ** order
            A0 = np.polyfit(x, adv_arr, 1)[1]
        elif len(dts_arr) == 1:
            A0 = float(adv_arr[0])
        if np.isfinite(A0) and np.isfinite(gr_val) and gr_val != 0.0:
            rel_err = abs(A0 - gr_val) / gr_val

    return {
        "planet": planet,
        "integrator": integrator,
        "rows": rows,
        "min_dt": dt_min,
        "max_dt": dt_max,
        "advance_mean": adv_mean,
        "advance_std": adv_std,
        "A0_extrapolated_check": A0,
        "relative_error_check": rel_err,
        "gr_rad_per_orbit": gr_val,
        "epsilon_used": eps_val,
        "periods": periods,
        "p_order": int(order),
    }


# -----------------------------------------------------------------------------
# Merge with R6 summary (optional but handy)
# -----------------------------------------------------------------------------

def merge_with_r6_summary(r7_rows_df: pd.DataFrame, r6_summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
      Combine R7 per-(planet, integrator) rollups with the corresponding
      values already computed by R6 in R6_perihelion_summary.csv.

      R6 columns are suffixed with "_r6" to avoid collisions.

    Returns:
      Merged dataframe, left-join on (planet, integrator).
    """
    if r7_rows_df is None or r7_rows_df.empty:
        return r7_rows_df
    if r6_summary_df is None or r6_summary_df.empty:
        return r7_rows_df

    for k in ["planet", "integrator"]:
        if k not in r7_rows_df.columns:
            r7_rows_df[k] = None
        if k not in r6_summary_df.columns:
            r6_summary_df[k] = None

    r6_cols = [c for c in r6_summary_df.columns if c not in ("planet", "integrator")]
    r6_summary_df = r6_summary_df.rename(columns={c: "{}_r6".format(c) for c in r6_cols})

    merged = r7_rows_df.merge(r6_summary_df, on=["planet", "integrator"], how="left")
    return merged


# =============================================================================
# Main
# =============================================================================

def main(argv=None):
    # Allow being called as main() from IDEs without CLI args
    if argv is None:
        argv = sys.argv[1:]

    ap = argparse.ArgumentParser(
        description="R7: consolidate R6 outputs into R7_summary.csv"
    )
    # Make --conf optional with a robust default
    ap.add_argument(
        "--conf",
        default=str(_default_conf_path()),
        help="Path to YAML config (defaults to ../config.yaml relative to this script)."
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="Print quick path and planet sanity info."
    )

    try:
        args = ap.parse_args(argv)
    except SystemExit as e:
        # Argparse uses SystemExit; show friendlier message in IDE runs
        if e.code == 2:
            print("[R7] Argument parse error. Try passing --conf C:\\path\\to\\config.yaml")
        raise

    # Validate conf path early with a clear message
    conf_path = Path(args.conf).expanduser().resolve()
    if not conf_path.exists():
        print("[R7] Config not found at:", conf_path)
        print("[R7] Fix by either:")
        print("  1) Creating the file there, or")
        print("  2) Running with: --conf C:\\path\\to\\config.yaml")
        return

    # ... keep the rest of your logic, but replace any previous parse with 'args'
    # Example of calling your existing flow:
    cfg = load_cfg(conf_path)
    out_dir = Path(cfg["paths"]["out_dir"]).expanduser().resolve()
    os.makedirs(out_dir, exist_ok=True)

    cfg_planets = list(cfg.get("r6", {}).get("planets", ["Mercury", "Venus", "Earth"]))
    cfg_planets_set = set(cfg_planets)
    planets = discover_planets(out_dir, allowed=cfg_planets_set) or cfg_planets
    integrators = list(cfg.get("r6", {}).get("integrators", ["vv", "rk4"]))

    if args.check:
        print("[R7] out_dir:", out_dir)
        print("[R7] cfg planets:", cfg_planets)
        print("[R7] discovered planets:", planets)
        print("[R7] integrators:", integrators)

    # Load R6 summary (optional)
    try:
        r6_summary_path = find_r6_summary(out_dir)
        r6_summary_df = pd.read_csv(r6_summary_path)
    except FileNotFoundError as e:
        print(str(e))
        r6_summary_df = pd.DataFrame()

    # Build rows
    rows = []
    for planet in planets:
        for integ in integrators:
            conv_csv = convergence_path(out_dir, planet, integ)
            if not conv_csv.exists():
                print("[R7] WARN Missing {}".format(conv_csv.name))
                continue
            try:
                df = pd.read_csv(conv_csv)
            except Exception as ex:
                print("[R7] ERROR reading {}: {}".format(conv_csv.name, ex))
                continue
            row = summarize_convergence_df(df, planet, integ)
            if row:
                rows.append(row)

    if not rows:
        print("[R7] nothing to summarize (no valid per-run CSVs found).")
        return

    r7_df = pd.DataFrame(rows).sort_values(["planet", "integrator"]).reset_index(drop=True)
    r7_merged = merge_with_r6_summary(r7_df, r6_summary_df)

    out_csv = out_dir / "R7_summary.csv"
    r7_merged.to_csv(out_csv, index=False)
    print("[R7] Wrote {} with {} rows".format(out_csv, len(r7_merged)))

def run(conf_path=None, **kwargs):
    argv = []
    if conf_path:
        argv += ["--conf", str(conf_path)]
    if "gamma" in kwargs and kwargs["gamma"] is not None:
        argv += ["--gamma", str(kwargs["gamma"])]
    if kwargs.get("check"):
        argv += ["--check"]
    return main(argv)

if __name__ == "__main__":
    main([])
