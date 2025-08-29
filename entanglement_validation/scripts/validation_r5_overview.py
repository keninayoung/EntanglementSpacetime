#!/usr/bin/env python3
# validation_r5_overview.py
# -----------------------------------------------------------------------------
# R5: Overview plots from entropy.csv and hawking_radiation.csv
# Produces two time series plots:
#   R5_entropy.png
#   R5_hawking.png
# under cfg["paths"]["out_dir"].
# -----------------------------------------------------------------------------

from __future__ import annotations
import os
from pathlib import Path
import argparse
import yaml
import pandas as pd

# Matplotlib in headless mode
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------- Config helpers ----------------------------

def _default_conf_path() -> Path:
    """
    Default to ../config.yaml relative to this file.
    Kept local so R5 does not depend on other modules.
    """
    here = Path(__file__).resolve()
    return (here.parent.parent / "config.yaml").resolve()

def _load_cfg(conf_path: Path) -> dict:
    with open(conf_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------- Core logic ----------------------------

def _plot_series(x, y, xlabel, ylabel, title, out_png):
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot(x, y, marker="o")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("[validation_r5] wrote {}".format(out_png))

def _r5_generate(cfg: dict):
    """
    Do the R5 work:
      - read cfg paths
      - load CSVs if present
      - make two small PNGs
    """
    # Resolve paths
    try:
        root = Path(cfg["paths"]["spacetime_outputs_dir"]).expanduser().resolve()
        out_dir = Path(cfg["paths"]["out_dir"]).expanduser().resolve()
    except KeyError as e:
        print("[validation_r5] ERROR missing config key:", e)
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Entropy
    ent_csv = root / "entropy.csv"
    if ent_csv.is_file():
        try:
            ent = pd.read_csv(ent_csv)
            if {"Step", "Entropy"}.issubset(ent.columns):
                _plot_series(
                    x=ent["Step"].values,
                    y=ent["Entropy"].values,
                    xlabel="Step",
                    ylabel="Entropy",
                    title="R5: Entropy over time",
                    out_png=str(out_dir / "R5_entropy.png"),
                )
            else:
                print("[validation_r5] entropy.csv missing required columns: Step, Entropy")
        except Exception as e:
            print("[validation_r5] ERROR reading {}: {}".format(ent_csv, e))
    else:
        print("[validation_r5] {} not found, skipping.".format(ent_csv.name))

    # 2) Hawking-like mutual information across horizon
    hk_csv = root / "hawking_radiation.csv"
    if hk_csv.is_file():
        try:
            hk = pd.read_csv(hk_csv)
            col_name = "MI Across Horizon"
            if {"Step", col_name}.issubset(hk.columns):
                _plot_series(
                    x=hk["Step"].values,
                    y=hk[col_name].values,
                    xlabel="Step",
                    ylabel="MI across horizon",
                    title="R5: Hawking-like MI over time",
                    out_png=str(out_dir / "R5_hawking.png"),
                )
            else:
                print("[validation_r5] hawking_radiation.csv missing required columns: Step, '{}'".format(col_name))
        except Exception as e:
            print("[validation_r5] ERROR reading {}: {}".format(hk_csv, e))
    else:
        print("[validation_r5] {} not found, skipping.".format(hk_csv.name))


# ---------------------------- Entry points ----------------------------

def main(argv=None):
    """
    CLI and orchestrator entry point. Accepts an argv list.
    """
    if argv is None:
        argv = []

    ap = argparse.ArgumentParser(
        description="validation_r5: overview plots from entropy.csv and hawking_radiation.csv",
        add_help=True
    )
    ap.add_argument(
        "--conf",
        default=str(_default_conf_path()),
        help="Path to YAML config. Defaults to ../config.yaml relative to this script."
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="Print sanity info and paths."
    )

    # Be forgiving about unknown flags (from an orchestrator)
    try:
        args, extras = ap.parse_known_args(argv)
        if extras:
            print("[validation_r5] INFO ignoring unknown args:", extras)
    except SystemExit as e:
        # 2 means argparse parse error; try with no args for IDE runs
        if e.code == 2:
            print("[validation_r5] WARN parse failed; retrying with no args")
            args = ap.parse_args([])
        else:
            raise

    conf_path = Path(args.conf).expanduser().resolve()
    if not conf_path.exists():
        print("[validation_r5] ERROR config not found:", conf_path)
        return

    cfg = _load_cfg(conf_path)

    if args.check:
        try:
            print("[validation_r5] out_dir:", cfg["paths"]["out_dir"])
            print("[validation_r5] spacetime_outputs_dir:", cfg["paths"]["spacetime_outputs_dir"])
        except KeyError:
            pass

    _r5_generate(cfg)
    print("[validation_r5] done.")

def run(conf_path=None, check=False):
    """
    Shim so run_validations.py can import and call run().
    """
    argv = []
    if conf_path:
        argv += ["--conf", str(conf_path)]
    if check:
        argv += ["--check"]
    return main(argv)

if __name__ == "__main__":
    # Allow running from IDE with no args
    main([])
