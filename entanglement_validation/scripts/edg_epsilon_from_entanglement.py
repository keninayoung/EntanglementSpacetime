# edg_epsilon_from_entanglement.py
# EDG demo: compute epsilon from entanglement outputs (Xi), then predict
# perihelion advances for one or more planets WITHOUT refitting.
#
# Why this exists:
#   R4/R6/R7 validate that EDG can reproduce classical tests (e.g., Mercury).
#   This script demonstrates the "quantum-informed predictor" loop:
#     1) Read curvature_lattice.csv produced by your PEPS pipeline.
#     2) Build an entanglement scalar Xi (e.g., mean absolute curvature at Step 0).
#     3) Calibrate k from Mercury: epsilon_mercury = k * Xi.
#     4) Predict epsilon for other planets: epsilon = k * Xi.
#     5) Integrate orbits with EDG acceleration; measure perihelion advance.
#     6) Compare to GR analytic value. No per-planet refits.
#
# Inputs:
#   - entanglement_validation/config.yaml (block: edg_demo)
#   - spacetime_outputs/curvature_lattice.csv (or path you specify)
#
# Outputs (under paths.out_dir, e.g., entanglement_validation/validation_outputs):
#   - EDG_epsilon_mapping.json   : Xi, k, epsilon_calibrated_mercury
#   - EDG_orbit_demo.csv         : per planet model vs GR
#   - EDG_orbit_demo.png         : bar chart of model vs GR
#
# Notes:
#   - ASCII only. No unicode.
#   - This script uses Velocity-Verlet and a robust periapsis detector
#     (sub-sampled via quadratic interpolation). For publication runs,
#     use large steps_per_orbit (e.g., 40k to 80k) and periods ~ 120.
#   - To keep this demo self-contained, we do not depend on R6 code here.

import os
import argparse
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------ physics helpers ------------------------

def delta_phi_gr(G, M, c, a, e):
    """GR perihelion advance per orbit (radians), weak-field."""
    return (6.0 * np.pi * G * M) / (a * (1.0 - e * e) * c * c)


def periapsis_state(mu, a, e):
    """Start at periapsis: r along +x, v along +y."""
    r_p = a * (1.0 - e)
    v_p = np.sqrt(mu * (1.0 + e) / (a * (1.0 - e)))
    r0 = np.array([r_p, 0.0], dtype=float)
    v0 = np.array([0.0, v_p], dtype=float)
    return r0, v0


def accel_edg(mu, c, r, v, epsilon):
    """
    EDG weak-field acceleration:
      a = -mu * rhat / r^2  -  (3 * epsilon * mu * h^2 / (c^2 r^4)) * rhat
    where h = |r x v| (scalar z-component magnitude in 2D).
    """
    x, y = r
    r2 = x * x + y * y
    rmag = np.sqrt(r2)
    if rmag == 0.0:
        return np.zeros(2, dtype=float)
    rhat = r / rmag
    a_vec = -mu * rhat / r2
    h = abs(r[0] * v[1] - r[1] * v[0])
    r4 = r2 * r2
    a_extra = -(3.0 * float(epsilon) * mu * (h * h) / (c * c * r4)) * rhat
    return a_vec + a_extra


def step_vv(r, v, dt, a_fn):
    """One Velocity-Verlet step."""
    a0 = a_fn(r, v)
    r_half = r + 0.5 * dt * v
    v_half = v + 0.5 * dt * a0
    a1 = a_fn(r_half + 0.5 * dt * v_half, v_half)
    v1 = v + 0.5 * dt * (a0 + a1)
    r1 = r + dt * v1
    return r1, v1


def integrate_orbit(mu, c, a, e, steps_per_orbit, periods, epsilon):
    """
    Integrate orbit using EDG acceleration and Velocity-Verlet.
    Returns unwrapped polar angle series, radius series, dt.
    """
    T = 2.0 * np.pi * np.sqrt(a ** 3 / mu)
    dt = T / float(steps_per_orbit)
    steps = int(max(1, periods * T / dt))

    r, v = periapsis_state(mu, a, e)
    theta = np.zeros(steps, dtype=float)
    radius = np.zeros(steps, dtype=float)

    def a_fn(rr, vv):
        return accel_edg(mu, c, rr, vv, epsilon)

    for n in range(steps):
        theta[n] = np.arctan2(r[1], r[0])
        radius[n] = np.hypot(r[0], r[1])
        r, v = step_vv(r, v, dt, a_fn)

    return np.unwrap(theta), radius, dt


def _quad_interp_min(y_minus, y0, y_plus):
    """
    Vertex of parabola through (-1, y_minus), (0, y0), (1, y_plus).
    Returns fractional offset x* in steps where minimum occurs.
    """
    denom = (y_minus - 2.0 * y0 + y_plus)
    if denom == 0.0:
        return 0.0
    return 0.5 * (y_minus - y_plus) / denom


def detect_perihelion_advance_subsample(theta_series, r_series, dt):
    """
    Detect perihelia by local minima of r. Refine time via quadratic fit,
    then interpolate the corresponding angle. Compute average increment
    between successive perihelion angles minus 2*pi baseline.
    """
    th = np.asarray(theta_series, dtype=float)
    rr = np.asarray(r_series, dtype=float)
    N = len(rr)

    # find local minima indices
    idxs = []
    for i in range(1, N - 1):
        if rr[i] <= rr[i - 1] and rr[i] <= rr[i + 1]:
            idxs.append(i)
    if len(idxs) < 2:
        return 0.0

    th = np.unwrap(th)
    peri_th = []
    for i in idxs:
        if i <= 0 or i >= N - 1:
            continue
        xstar = _quad_interp_min(rr[i - 1], rr[i], rr[i + 1])  # fractional step offset
        if xstar >= 0.0:
            frac = min(1.0, max(0.0, xstar))
            th_ref = (1.0 - frac) * th[i] + frac * th[i + 1]
        else:
            frac = min(1.0, max(0.0, 1.0 + xstar))
            th_ref = (1.0 - frac) * th[i - 1] + frac * th[i]
        peri_th.append(th_ref)

    if len(peri_th) < 2:
        return 0.0

    deltas = np.diff(np.array(peri_th))
    advance = float(np.mean(np.maximum(0.0, deltas - 2.0 * np.pi)))
    return advance


# ------------------------ entanglement -> epsilon ------------------------

def load_xi(curv_file, step0_col, mode):
    """
    Load curvature_lattice.csv and compute Xi from the Step 0 column.
    mode: "mean_abs", "mean", or "rms".
    Returns (Xi_value, used_column_name).
    """
    if not os.path.exists(curv_file):
        raise FileNotFoundError("Could not find curvature file: {}".format(curv_file))

    df = pd.read_csv(curv_file)

    # auto-detect "Step 0" if not provided or missing
    col = step0_col if step0_col and step0_col in df.columns else None
    if col is None:
        for c in df.columns:
            if isinstance(c, str) and c.strip().lower().startswith("step 0"):
                col = c
                break
    if col is None:
        if len(df.columns) < 2:
            raise ValueError("curvature file must have at least two columns")
        col = df.columns[1]

    vals = df[col].values.astype(float)

    if mode == "mean_abs":
        Xi = float(np.mean(np.abs(vals)))
    elif mode == "mean":
        Xi = float(np.mean(vals))
    elif mode == "rms":
        Xi = float(np.sqrt(np.mean(vals * vals)))
    else:
        raise ValueError("Unknown xi_mode: {}".format(mode))

    return Xi, col


# ------------------------------- driver --------------------------------

def run(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    out_dir = cfg["paths"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    edg = cfg.get("edg_demo", {})
    curv_file = edg.get("curvature_file", "spacetime_outputs/curvature_lattice.csv")
    step0_column = edg.get("step0_column", "Step 0")
    xi_mode = edg.get("xi_mode", "mean_abs")
    planets = list(edg.get("planets", ["Mercury", "Venus", "Earth"]))
    epsilon_calibrated_mercury = float(edg.get("epsilon_calibrated_mercury", 1.0))

    steps_per_orbit = int(edg.get("steps_per_orbit", 40000))
    periods = int(edg.get("periods", 120))

    consts = edg.get("constants", {})
    G = float(consts.get("G", 6.67430e-11))
    M_sun = float(consts.get("M_sun", 1.98847e30))
    c = float(consts.get("c", 2.99792458e8))
    mu = G * M_sun

    planet_defs = edg.get("constants_planets", {})
    if "Mercury" not in planet_defs:
        raise ValueError("constants_planets must include Mercury for calibration")

    # 1) Compute Xi from entanglement outputs
    Xi, used_col = load_xi(curv_file, step0_column, xi_mode)
    print("[EDG] Xi from {} column '{}' with mode '{}': {:.6e}".format(curv_file, used_col, xi_mode, Xi))

    # 2) Calibrate k from Mercury: epsilon_mercury = k * Xi  =>  k = epsilon_mercury / Xi
    if Xi == 0.0:
        raise ValueError("Xi evaluated to zero; cannot calibrate k")
    k = epsilon_calibrated_mercury / Xi
    print("[EDG] Calibrated k using Mercury: k = eps_mercury / Xi = {:.6e}".format(k))

    # 3) Predict epsilon for each planet and measure perihelion advance
    rows = []
    for planet in planets:
        if planet not in planet_defs:
            print("[EDG] Skipping planet not in constants_planets: {}".format(planet))
            continue

        a = float(planet_defs[planet]["a"])
        e = float(planet_defs[planet]["e"])
        epsilon_pred = float(k * Xi)

        # integrate and measure advance
        theta, radius, dt = integrate_orbit(mu, c, a, e, steps_per_orbit, periods, epsilon_pred)
        adv_model = detect_perihelion_advance_subsample(theta, radius, dt)
        adv_gr = delta_phi_gr(G, M_sun, c, a, e)
        rel_err = abs(adv_model - adv_gr) / adv_gr if adv_gr != 0.0 else np.nan

        rows.append({
            "planet": planet,
            "Xi": float(Xi),
            "k": float(k),
            "epsilon_pred": float(epsilon_pred),
            "advance_model_rad_per_orbit": float(adv_model),
            "advance_GR_rad_per_orbit": float(adv_gr),
            "relative_error": float(rel_err),
            "steps_per_orbit": int(steps_per_orbit),
            "periods": int(periods),
            "dt": float(dt)
        })

        print("[EDG] {}: eps_pred={:.6e}, model={:.6e}, GR={:.6e}, rel_err={:.3e}".format(
            planet, epsilon_pred, adv_model, adv_gr, rel_err))

    # 4) Write mapping json and results csv
    map_path = os.path.join(out_dir, "EDG_epsilon_mapping.json")
    with open(map_path, "w") as f:
        json.dump({
            "Xi": float(Xi),
            "xi_mode": xi_mode,
            "used_column": used_col,
            "epsilon_calibrated_mercury": float(epsilon_calibrated_mercury),
            "k_from_mercury": float(k),
        }, f, indent=2)
    print("[EDG] Wrote {}".format(map_path))

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "EDG_orbit_demo.csv")
    df.to_csv(csv_path, index=False)
    print("[EDG] Wrote {}".format(csv_path))

    # 5) Simple model vs GR bar plot
    if len(rows) > 0:
        planets_order = [r["planet"] for r in rows]
        model_vals = [r["advance_model_rad_per_orbit"] for r in rows]
        gr_vals = [r["advance_GR_rad_per_orbit"] for r in rows]

        x = np.arange(len(planets_order))
        w = 0.35
        plt.figure(figsize=(8, 4))
        plt.bar(x - w / 2.0, model_vals, width=w, label="Model (EDG, epsilon from Xi)")
        plt.bar(x + w / 2.0, gr_vals,    width=w, label="GR")
        plt.xticks(x, planets_order)
        plt.ylabel("advance (rad/orbit)")
        plt.title("EDG perihelion: epsilon predicted from Xi")
        plt.grid(True, axis="y")
        plt.legend(loc="best")
        png_path = os.path.join(out_dir, "EDG_orbit_demo.png")
        plt.savefig(png_path, dpi=160, bbox_inches="tight")
        print("[EDG] Wrote {}".format(png_path))


def main():
    ap = argparse.ArgumentParser()
    default_conf = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
    ap.add_argument("--conf", default=default_conf, help="Path to YAML config (default: entanglement_validation/config.yaml)")
    args = ap.parse_args()
    run(args.conf)


if __name__ == "__main__":
    main()
