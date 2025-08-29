# validation_r9_ppn_cross_checks.py
# Cross-validate with classic GR tests using PPN gamma.
# Produces light bending at solar limb and Shapiro delay estimates.
# ASCII-only, IDE-friendly.

from __future__ import annotations
import sys
from pathlib import Path
import argparse
import math
import pandas as pd

from entanglement_validation.scripts.physics_common import (
    load_cfg, default_conf_path, ensure_out_dir,
    G_SI, M_SUN_SI, C_SI, R_SUN_SI, AU_SI, ARCSEC_PER_RAD,
    arcsec_from_radians
)


def light_bending_ppn_gamma(G, M, c, b, gamma_ppn=1.0):
    """
    Total deflection angle in radians for a light ray with impact parameter b:
      alpha = 4*G*M / (c^2 * b) * (gamma_ppn + 1) / 2
    For GR, gamma_ppn = 1, so alpha = 4*G*M / (c^2 * b).
    The factorization here is written to make the gamma dependence explicit.
    """
    return 2.0 * (G * M) / (c * c * b) * (gamma_ppn + 1.0)

def shapiro_delay_ppn_gamma(G, M, c, r1, r2, R, gamma_ppn=1.0):
    """
    One-way Shapiro delay (seconds) in the standard log form for a point mass:
      Delta t = (1 + gamma) * G*M / c^3 * ln( (r1 + r2 + R) / (r1 + r2 - R) )
    r1 = distance from mass to transmitter (m)
    r2 = distance from mass to receiver (m)
    R  = Euclidean separation transmitter-receiver (m)
    For superior conjunction and near-collinearity, R ~ r1 + r2.
    This function does not include geometric terms unrelated to GR time delay.
    """
    num = (r1 + r2 + R)
    den = max(1e-30, (r1 + r2 - R))  # guard zero in log
    return ((1.0 + gamma_ppn) * G * M / (c ** 3)) * math.log(num / den)

def main(argv=None):
    if argv is None:
        argv = []

    ap = argparse.ArgumentParser(description="R9 PPN cross-checks: light bending and Shapiro delay")
    ap.add_argument("--conf", default=str(default_conf_path(Path(__file__).resolve())),
                    help="Path to YAML config. Defaults to ../config.yaml relative to this script.")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="PPN gamma to evaluate. Default 1.0 (GR).")
    ap.add_argument("--check", action="store_true", help="Print sanity info.")
    args = ap.parse_args(argv)

    cfg = load_cfg(Path(args.conf))
    out_dir = ensure_out_dir(cfg)
    gamma = float(args.gamma)

    # 1) Light bending at solar limb
    alpha_rad = light_bending_ppn_gamma(G_SI, M_SUN_SI, C_SI, R_SUN_SI, gamma_ppn=gamma)
    alpha_arcsec = arcsec_from_radians(alpha_rad)

    # 2) Shapiro delay for an Earth-spacecraft link near solar conjunction.
    #    Simple configuration: r1 = 1 AU (Earth), r2 = 1 AU (spacecraft),
    #    R ~ r1 + r2 - small (near-collinear). For a worst case, set R very close
    #    to r1 + r2 but not equal to avoid log singularity.
    r1 = AU_SI
    r2 = AU_SI
    # Choose R slightly smaller than r1 + r2 by ~ solar radius projected
    R = r1 + r2 - R_SUN_SI
    dt_shapiro = shapiro_delay_ppn_gamma(G_SI, M_SUN_SI, C_SI, r1, r2, R, gamma_ppn=gamma)

    # Save a small CSV for the record
    rows = [dict(
        gamma_ppn=gamma,
        light_bending_rad=alpha_rad,
        light_bending_arcsec=alpha_arcsec,
        shapiro_delay_seconds=dt_shapiro,
        config_r1_m=r1,
        config_r2_m=r2,
        config_R_m=R,
    )]
    df = pd.DataFrame(rows)
    out_csv = out_dir / "R9_ppn_checks.csv"
    df.to_csv(out_csv, index=False)
    print("[R9] wrote {}".format(out_csv))

    if args.check:
        print("[R9] light bending (arcsec):", alpha_arcsec)
        print("[R9] shapiro delay (microseconds):", dt_shapiro * 1e6)

def run(conf_path=None, gamma=None, check=False):
    """
    Thin adapter so run_validations.py can import and call run().

    Parameters
    ----------
    conf_path : str or Path or None
        Optional path to config.yaml. If None, the script default is used.
    gamma : float or None
        PPN gamma to evaluate. If None, script default is used.
    check : bool
        If True, print sanity info.
    """
    argv = []
    if conf_path:
        argv += ["--conf", str(conf_path)]
    if gamma is not None:
        argv += ["--gamma", str(float(gamma))]
    if check:
        argv += ["--check"]

    return main(argv)


if __name__ == "__main__":
    main([])

