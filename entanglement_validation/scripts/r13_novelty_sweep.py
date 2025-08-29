#!/usr/bin/env python3
"""
r13_novelty_sweep.py

Purpose
  Systematically sweep strong-field knobs and EHT precision to find regions
  that favor EDG over GR (or vice versa).

What it does
  - Loads your external R13 CSV (or minimal default) once
  - For each combination of:
        target in {m87, sgrA}
        kappa in a list (e.g. 0.05, 0.1, 0.2)
        p      in a list (e.g. 1, 2, 4)
        EHT sigma in a list (e.g. 2.0, 1.5, 1.0, 0.75, 0.5 microas)
    builds a new ETH row and runs the same posterior grid as R13+
  - Computes:
        Lq95 (upper limit excluding zero)
        DeltaAIC, DeltaBIC, DeltaLogZ (EDG minus GR)
        epsilon_best, Lq_best, chi2_best
  - Saves a single CSV: R13p_novelty_sweep.csv in validation_outputs/

Use
  python -m entanglement_validation.scripts.r13_novelty_sweep --conf .../config.yaml \
      --ext_csv .../external_datasets/r13_external_from_R12.csv \
      --targets m87,sgrA \
      --kappa_list 0.05,0.1,0.2 \
      --p_list 1,2,4 \
      --sigma_list 2.0,1.5,1.0,0.75,0.5 \
      --eps_grid 0.98,1.02,201 \
      --Lq_grid 1e4,5e7,201

Notes
  - ASCII only, headless-safe.
  - This reuses functions from validation_r13p_joint_observational_fit.py
    so keep that file as provided earlier.
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Reuse R13+p internals
from entanglement_validation.scripts.validation_r13p_joint_observational_fit import (
    _load_or_minimal, build_eht_row, grid_posterior, marginals_from_grid,
    eval_model_metrics, upper_limit_from_grid, _parse_grid, _fmt_float
)
from entanglement_validation.scripts.physics_common import (
    load_cfg, ensure_out_dir
)

def _as_list_floats(s):
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]

def _targets_list(s):
    return [t.strip() for t in str(s).split(",") if t.strip()]

def main(argv=None):
    if argv is None:
        argv = []

    ap = argparse.ArgumentParser(
        description="R13 novelty sweep over (target, kappa, p, EHT sigma)."
    )
    ap.add_argument("--conf", required=True, help="Path to entanglement_validation/config.yaml")
    ap.add_argument("--ext_csv", default=None, help="Path to r13_external_from_R12.csv (or leave None to use minimal).")

    ap.add_argument("--targets", default="m87", help="m87,sgrA or either one.")
    ap.add_argument("--kappa_list", default="0.1", help="Comma list of kappa values.")
    ap.add_argument("--p_list", default="2.0", help="Comma list of p powers.")
    ap.add_argument("--sigma_list", default="2.0,1.5,1.0,0.75,0.5", help="Comma list of EHT sigmas (microas).")

    ap.add_argument("--eps_grid", default="0.98,1.02,201", help="epsilon grid start,stop,num")
    ap.add_argument("--Lq_grid", default="1e4,5e7,201", help="L_q grid start,stop,num (meters)")

    # Reasonable GR baselines and masses for EHT targets
    ap.add_argument("--m87_DGR", type=float, default=42.0)
    ap.add_argument("--m87_mass", type=float, default=6.5e9)
    ap.add_argument("--sgrA_DGR", type=float, default=51.8)
    ap.add_argument("--sgrA_mass", type=float, default=4.1e6)

    args = ap.parse_args(argv)

    cfg = load_cfg(Path(args.conf))
    out_dir = ensure_out_dir(cfg)

    # Load base ext data
    base_df = _load_or_minimal(args.ext_csv)
    if base_df.empty:
        print("[R13Sweep] ERROR: base external dataset is empty.")
        return

    eps_grid = _parse_grid(args.eps_grid)
    Lq_grid  = _parse_grid(args.Lq_grid)

    targets = _targets_list(args.targets)
    kappa_vals = _as_list_floats(args.kappa_list)
    p_vals = _as_list_floats(args.p_list)
    sigmas = _as_list_floats(args.sigma_list)

    results = []

    for target in targets:
        target_lower = target.lower()
        if target_lower not in ("m87", "sgra"):
            print("[R13Sweep] WARN skipping unknown target:", target)
            continue

        if target_lower == "m87":
            DGR = float(args.m87_DGR)
            mass = float(args.m87_mass)
        else:
            DGR = float(args.sgrA_DGR)
            mass = float(args.sgrA_mass)

        for kappa in kappa_vals:
            for p in p_vals:
                for sig in sigmas:
                    # Build a working copy of the dataset with EHT row for this combo
                    df = base_df.copy(deep=True)
                    eht_row = build_eht_row(
                        target=target_lower,
                        D_GR_microas=DGR,
                        sigma_microas=float(sig),
                        mass_solar=mass,
                        kappa=float(kappa),
                        p_power=float(p)
                    )
                    df = pd.concat([df, pd.DataFrame([eht_row])], ignore_index=True)

                    # Posterior and stats
                    post = grid_posterior(df, eps_grid, Lq_grid)
                    summary, m_eps, m_Lq, Sigma = marginals_from_grid(post)
                    stats = eval_model_metrics(df, post, Sigma, eps_grid, Lq_grid)

                    # Credible summaries
                    Lq95 = upper_limit_from_grid(post, cl=0.95, exclude_zero=True)

                    # Best fit
                    best_idx = int(post["posterior_norm"].idxmax())
                    best = post.iloc[best_idx]

                    results.append(dict(
                        target=target_lower,
                        kappa=float(kappa),
                        p=float(p),
                        eht_sigma=float(sig),
                        epsilon_best=float(best["epsilon"]),
                        Lq_best=float(best["L_q"]),
                        chi2_best=float(best["chi2"]),
                        Lq95=float(Lq95),
                        DeltaAIC=float(stats["DeltaAIC"]),
                        DeltaBIC=float(stats["DeltaBIC"]),
                        DeltaLogZ=float(stats["DeltaLogZ"])
                    ))

                    print("[R13Sweep] target={} kappa={} p={} sigma={}  ->  "
                          "Lq95={}  DeltaLogZ={}  eps*={} Lq*={}".format(
                              target_lower, _fmt_float(kappa), _fmt_float(p), _fmt_float(sig),
                              _fmt_float(Lq95), _fmt_float(stats["DeltaLogZ"]),
                              _fmt_float(best["epsilon"]), _fmt_float(best["L_q"])
                          ))

    out_csv = out_dir / "R13p_novelty_sweep.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print("[R13Sweep] wrote {}".format(out_csv))


def run(conf_path=None, ext_csv=None, targets="m87",
        kappa_list="0.1", p_list="2.0", sigma_list="2.0,1.5,1.0,0.75,0.5",
        eps_grid="0.98,1.02,201", Lq_grid="1e4,5e7,201",
        m87_DGR=42.0, m87_mass=6.5e9, sgrA_DGR=51.8, sgrA_mass=4.1e6):
    argv = ["--conf", str(conf_path)]
    if ext_csv:
        argv += ["--ext_csv", str(ext_csv)]
    argv += ["--targets", str(targets),
             "--kappa_list", str(kappa_list),
             "--p_list", str(p_list),
             "--sigma_list", str(sigma_list),
             "--eps_grid", str(eps_grid),
             "--Lq_grid", str(Lq_grid),
             "--m87_DGR", str(float(m87_DGR)),
             "--m87_mass", str(float(m87_mass)),
             "--sgrA_DGR", str(float(sgrA_DGR)),
             "--sgrA_mass", str(float(sgrA_mass))]
    return main(argv)


if __name__ == "__main__":
    main()
