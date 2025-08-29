#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# r13_build_from_r12.py
#
# Build an external CSV for R13 from R12 results, automatically pulling
# per-planet bootstrap sigmas from R8_perihelion_bootstrap.csv when available.
#
# Outputs:
#   r13_external_from_R12.csv  (path controlled by --csv_out)
#
# Notes:
#   - If R8 file is missing, falls back to max(|A0-GR|, rel_err*GR, sigma_floor).
#   - ASCII only; robust CSV reading; prints columns it finds.
# -----------------------------------------------------------------------------

import os, json, argparse
import pandas as pd
import numpy as np

def _safe_read_csv(path):
    try:
        if os.path.isfile(path):
            df = pd.read_csv(path)
            print("[R13C] INFO loaded {} with columns: {}".format(path, list(df.columns)))
            return df
    except Exception as e:
        print("[R13C] WARN could not read {}: {}".format(path, e))
    return None

def _find_bootstrap_sigmas(out_dir):
    """
    Look first in eps=1 folder, then in any R12_eps_* folder.
    Expect a file named R8_perihelion_bootstrap.csv with columns:
      planet, sigma_rad_per_orbit   (preferred)
    or  planet, A0_boot_std         (accepted)
    or  planet, sigma               (fallback)
    Returns dict: {planet: sigma}
    """
    lookup = {}

    # Preferred path: eps = 1
    eps1 = os.path.join(out_dir, "R12_eps_1.000000000", "R8_perihelion_bootstrap.csv")
    df = _safe_read_csv(eps1)
    if df is None:
        # Try any R12_eps_* folder
        for d in os.listdir(out_dir):
            cand = os.path.join(out_dir, d, "R8_perihelion_bootstrap.csv")
            if os.path.isfile(cand):
                df = _safe_read_csv(cand)
                if df is not None:
                    break

    if df is not None:
        for _, row in df.iterrows():
            p = str(row.get("planet", "")).strip()
            s = None
            # accept multiple naming conventions
            if "sigma_rad_per_orbit" in df.columns:
                s = row.get("sigma_rad_per_orbit", None)
            elif "A0_boot_std" in df.columns:
                s = row.get("A0_boot_std", None)
            elif "sigma" in df.columns:
                s = row.get("sigma", None)
            if p and s is not None and np.isfinite(s) and float(s) > 0.0:
                lookup[p] = float(s)
                print("[R13C] INFO bootstrap sigma for {} = {:.3e} rad/orbit".format(p, float(s)))

    if not lookup:
        print("[R13C] INFO no usable R8_perihelion_bootstrap.csv found; will use fallbacks.")
    return lookup

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="validation_outputs directory")
    ap.add_argument("--csv_out", default="r13_external_from_R12.csv")
    ap.add_argument("--use_planets", default="Mercury,Venus,Earth")
    ap.add_argument("--eps_target", type=float, default=1.0, help="which epsilon row to use from R12 consensus")
    ap.add_argument("--sigma_floor", type=float, default=1e-10, help="absolute floor for perihelion sigma (rad/orbit)")
    args = ap.parse_args(argv)

    out_dir = os.path.abspath(args.out_dir)
    planets = [p.strip() for p in args.use_planets.split(",") if p.strip()]

    # R12 consensus across epsilons
    r12_path = os.path.join(out_dir, "R12_A0_vs_epsilon_consensus.csv")
    if not os.path.isfile(r12_path):
        raise SystemExit("Missing {}".format(r12_path))
    r12 = pd.read_csv(r12_path)

    # R6 summary at eps=1 for GR baseline in meta, if available
    r6_path = os.path.join(out_dir, "R12_eps_1.000000000", "R6_perihelion_summary_eps_1.000000000.csv")
    r6 = _safe_read_csv(r6_path)

    # Pull bootstrap sigmas if present
    bs_lookup = _find_bootstrap_sigmas(out_dir)

    rows = []
    for p in planets:
        r = r12[(r12["planet"] == p) & (np.isclose(r12["epsilon"], args.eps_target))]
        if r.empty:
            print("[R13C] WARN: no row for planet={} at epsilon={}".format(p, args.eps_target))
            continue
        A0 = float(r["A0_consensus"].iloc[0])

        # GR target for meta
        if r6 is not None and "gr_rad_per_orbit" in r6.columns:
            r6p = r6[r6["planet"] == p]
            if not r6p.empty:
                gr = float(r6p["gr_rad_per_orbit"].iloc[0])
                rel_err = float(r6p["relative_error"].iloc[0]) if "relative_error" in r6p.columns else 0.0
            else:
                gr, rel_err = A0, 0.0
        else:
            gr, rel_err = A0, 0.0

        # Choose sigma
        if p in bs_lookup:
            sigma = float(bs_lookup[p])
        else:
            sigma = max(abs(A0 - gr), abs(rel_err * gr), float(args.sigma_floor))

        print("[R13C] {:>8s}  A0={}  GR={}  sigma={}".format(
            p, "{:.6e}".format(A0), "{:.6e}".format(gr), "{:.6e}".format(sigma)))

        rows.append(dict(
            dataset_id=f"{p.lower()}_peri_r12",
            dataset_type="perihelion",
            observable="A0",
            value=A0,
            sigma=sigma,
            units="rad/orbit",
            planet=p,
            meta_json=json.dumps({"gr_rad_per_orbit": gr})
        ))

    # Cassini gamma and Shapiro to anchor epsilon
    rows.append(dict(
        dataset_id="cassini_gamma",
        dataset_type="gamma",
        observable="gamma",
        value=1.000021,
        sigma=2.3e-05,
        units="dimensionless",
        planet="",
        meta_json="{}"
    ))
    rows.append(dict(
        dataset_id="cassini_shapiro",
        dataset_type="shapiro",
        observable="delay",
        value=6.655444480760713e-05,
        sigma=1.53e-09,
        units="seconds",
        planet="",
        meta_json=json.dumps({"gr_pred_seconds": 6.655444480760713e-05})
    ))

    df = pd.DataFrame(rows)
    out_csv = os.path.abspath(args.csv_out)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("[R13C] wrote", out_csv)

def run(conf_path=None, csv_out=None, use_planets=None, eps_target=None, sigma_floor=None):
    # Determine validation_outputs from conf.yaml location
    if conf_path is None:
        raise SystemExit("Need conf_path to locate validation_outputs")
    # The physics_common helpers already make validation_outputs; but here we
    # expect the user to pass the explicit path. Keep it simple:
    out_dir = os.path.join(os.path.dirname(conf_path), "validation_outputs")
    argv = ["--out_dir", out_dir]
    if csv_out:
        argv += ["--csv_out", str(csv_out)]
    if use_planets:
        argv += ["--use_planets", str(use_planets)]
    if eps_target is not None:
        argv += ["--eps_target", str(float(eps_target))]
    if sigma_floor is not None:
        argv += ["--sigma_floor", str(float(sigma_floor))]
    return main(argv)

if __name__ == "__main__":
    main()
