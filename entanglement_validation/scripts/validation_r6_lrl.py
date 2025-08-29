#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# R6: High-precision perihelion benchmark for EDG vs GR.
#
# What is new in this version (read this):
#   1) Robust perihelion advance estimator:
#        - Detect perihelia from the radius series (min-with-gap).
#        - Sample the LRL angle ONLY at those perihelia.
#        - Unwrap, take successive differences, and subtract the nearest
#          multiple of 2*pi (modulo baseline), leaving only the tiny drift.
#        - Average over the requested perihelion window.
#      This eliminates the failure mode where all methods returned ~ -2*pi.
#
#   2) We still compute legacy detectors for logging:
#        - "sub": radius-minima subsampling method (can fail to -2*pi)
#        - "lrl": windowed difference of LRL angle over one orbit (can fail)
#        - "fit": linear fit of LRL angle vs sample index (can fail)
#      But the final "advance_rad_per_orbit" is chosen from the robust
#      perihelion-anchored estimator when available.
#
#   3) Filename suffix support for ALL outputs:
#        R6_perihelion_convergence_<Planet>_<Integrator><sfx>.csv
#        R6_perihelion_convergence_<Planet>_<Integrator><sfx>.png
#        R6_perihelion_summary<sfx>.csv
#      where sfx is like "_eps_1.000000000".
#
#   4) Performance knobs:
#        r6.max_total_steps  (int) clamp total steps per job; 0 or missing = off
#        r6.store_stride     (int) store every Nth sample; integrate all steps
#
#   5) Stable A0 extrapolation for plotting and summary (no polyfit warnings).
#
# Config keys used (YAML):
#   paths.out_dir
#   r6.planets, r6.integrators, r6.steps_per_orbit_list, r6.periods
#   r6.use_perihelia_from, r6.use_perihelia_to
#   r6.epsilon
#   r6.filename_suffix        # optional, string without leading underscore
#   r6.max_total_steps        # optional, int
#   r6.store_stride           # optional, int
#   r6.constants.{G,M_sun,c}
#   r6.constants_planets.<Planet>.{a,e}
#
# ASCII-only.
# -----------------------------------------------------------------------------

import os
import sys
import time
import math
import argparse
import subprocess

# Safety: cap BLAS threads unless user overrides
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import yaml
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------- Utility: suffix, stable fit ----------------------------

def _suffix_from_cfg(cfg_dict):
    """
    Returns "" or a suffix like "_eps_1.000000000" based on cfg["r6"]["filename_suffix"].
    """
    try:
        sfx_raw = str(cfg_dict.get("r6", {}).get("filename_suffix", "")).strip()
    except Exception:
        sfx_raw = ""
    return ("_" + sfx_raw) if sfx_raw else ""


def _stable_fit_intercept(x_raw, y_raw):
    """
    Fit y = a*x + b in a numerically stable way and return the intercept b at x=0.
    """
    x = np.asarray(x_raw, dtype=float).ravel()
    y = np.asarray(y_raw, dtype=float).ravel()
    n = x.size
    if n == 0:
        return float("nan")
    if n == 1:
        return float(y[0])

    xm = float(np.mean(x))
    xc = x - xm
    xs = float(np.max(np.abs(xc))) or 1.0
    X = np.column_stack([xc / xs, np.ones_like(x)])
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        a_scaled, b_scaled = float(beta[0]), float(beta[1])
    except Exception:
        # tiny ridge if lstsq fails
        XtX = X.T @ X
        XtY = X.T @ y
        lam = 1e-12
        beta = np.linalg.solve(XtX + lam * np.eye(2), XtY)
        a_scaled, b_scaled = float(beta[0]), float(beta[1])

    # intercept at x=0
    b0 = b_scaled - a_scaled * (xm / xs)
    return float(b0)


# ---------------------------- Physics helpers ----------------------------

def delta_phi_gr(G, M, c, a, e):
    # GR perihelion advance per orbit (radians), weak-field.
    return (6.0 * math.pi * G * M) / (a * (1.0 - e * e) * c * c)


def periapsis_state(mu, a, e):
    # Start at periapsis: position +x, velocity +y.
    r_p = a * (1.0 - e)
    v_p = math.sqrt(mu * (1.0 + e) / (a * (1.0 - e)))
    r0 = np.array([r_p, 0.0], dtype=float)
    v0 = np.array([0.0, v_p], dtype=float)
    return r0, v0


def accel_edg(mu, c, r, v, epsilon):
    # EDG weak-field radial correction added to Newtonian.
    # a = -mu rhat / r^2  - (3 eps mu h^2 / (c^2 r^4)) rhat
    x, y = r
    r2 = x * x + y * y
    rmag = math.sqrt(r2)
    if rmag == 0.0:
        return np.zeros(2, dtype=float)
    rhat = r / rmag
    aN = -mu * rhat / r2
    h = abs(r[0] * v[1] - r[1] * v[0])
    r4 = r2 * r2
    aE = -(3.0 * float(epsilon) * mu * (h * h) / (c * c * r4)) * rhat
    return aN + aE


# ---------------------------- Integrators ----------------------------

def vv_step(r, v, dt, a_fn):
    a0 = a_fn(r, v)
    r_half = r + 0.5 * dt * v
    v_half = v + 0.5 * dt * a0
    a1 = a_fn(r_half + 0.5 * dt * v_half, v_half)
    v1 = v + 0.5 * dt * (a0 + a1)
    r1 = r + dt * v1
    return r1, v1


def rk4_step(r, v, dt, a_fn):
    def f(state):
        rr = state[:2]
        vv = state[2:]
        a = a_fn(rr, vv)
        return np.array([vv[0], vv[1], a[0], a[1]], dtype=float)

    y = np.array([r[0], r[1], v[0], v[1]], dtype=float)
    k1 = f(y)
    k2 = f(y + 0.5 * dt * k1)
    k3 = f(y + 0.5 * dt * k2)
    k4 = f(y + dt * k3)
    y_next = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return y_next[:2], y_next[2:]


def get_stepper(name):
    if name == "vv":
        return vv_step
    if name == "rk4":
        return rk4_step
    raise ValueError("Unknown integrator: {}".format(name))


# ---------------------------- Perihelion detectors (legacy + robust) ----------------------------

def detect_perihelion_advance_subsample(theta_series, radius_series, steps_per_orbit):
    """
    Legacy detector: looks for radius minima with a gap and does quadratic
    interpolation of the angle at the min. This can fail to ~ -2*pi if minima
    are missed. Kept for diagnostics.
    """
    th = np.unwrap(np.asarray(theta_series, dtype=float))
    rr = np.asarray(radius_series, dtype=float)
    N = len(rr)
    if N < 3:
        return 0.0

    min_gap = max(1, int(0.6 * int(steps_per_orbit)))

    def quad_interp_min(y_minus, y0, y_plus):
        denom = (y_minus - 2.0 * y0 + y_plus)
        if denom == 0.0:
            return 0.0
        return 0.5 * (y_minus - y_plus) / denom

    peri_angles = []
    last_idx = -10**9

    for i in range(1, N - 1):
        if i - last_idx < min_gap:
            continue
        if rr[i] <= rr[i - 1] and rr[i] <= rr[i + 1]:
            xstar = quad_interp_min(rr[i - 1], rr[i], rr[i + 1])
            if xstar >= 0.0:
                frac = min(1.0, max(0.0, xstar))
                ang = (1.0 - frac) * th[i] + frac * th[i + 1]
            else:
                frac = min(1.0, max(0.0, 1.0 + xstar))
                ang = (1.0 - frac) * th[i - 1] + frac * th[i]
            peri_angles.append(ang)
            last_idx = i

    if len(peri_angles) < 2:
        return 0.0

    peri_angles = np.unwrap(np.array(peri_angles, dtype=float))
    deltas = np.diff(peri_angles)
    mean_delta = float(np.mean(deltas))
    baseline = math.copysign(2.0 * math.pi, mean_delta if mean_delta != 0.0 else 1.0)
    return float(np.mean(deltas - baseline))


def measure_precession_from_A(A_thetas, steps_per_orbit, use_from, use_to):
    """
    Legacy: difference of LRL angle separated by exactly one orbit worth of
    stored samples. Can fail if store_stride or partial orbits misalign window.
    """
    A = np.unwrap(np.asarray(A_thetas, dtype=float))
    N = len(A)
    if steps_per_orbit <= 0:
        return 0.0
    samples = []
    for k in range(use_from, use_to):
        i = k * steps_per_orbit
        j = i + steps_per_orbit
        if j < N:
            samples.append(A[j] - A[i])
    if not samples:
        return 0.0
    return float(np.mean(np.array(samples) - 2.0 * math.pi))


def estimate_precession_linear_fit(A_unwrapped, steps_per_orbit_eff, use_from, use_to):
    """
    Legacy: linear fit of unwrapped LRL angle A vs stored-sample index.
    Returns slope * steps_per_orbit_eff - 2*pi. Can fail to -2*pi if slope~0.
    """
    A = np.asarray(A_unwrapped, dtype=float).ravel()
    n = A.size
    if n < 2 or steps_per_orbit_eff <= 0:
        return 0.0

    i0 = max(0, int(use_from * steps_per_orbit_eff))
    i1 = min(n, int(use_to   * steps_per_orbit_eff))
    if i1 - i0 < 2:
        i0, i1 = 0, n

    x = np.arange(i0, i1, dtype=float)
    y = A[i0:i1]
    if y.size < 2:
        return 0.0

    xbar = float(np.mean(x))
    xc = x - xbar
    denom = float(np.dot(xc, xc))
    if denom <= 0.0:
        return 0.0
    a = float(np.dot(xc, y - np.mean(y)) / denom)   # rad per stored sample
    adv = a * float(steps_per_orbit_eff) - 2.0 * math.pi
    return float(adv)


def _peri_indices_from_radius(radius_series, steps_per_orbit_eff):
    """
    Return indices of perihelia using a min-with-gap rule on the radius series.
    """
    rr = np.asarray(radius_series, dtype=float)
    n = rr.size
    if n < 3:
        return []
    gap = max(1, int(0.6 * int(steps_per_orbit_eff)))
    idxs = []
    last = -10**9
    for i in range(1, n - 1):
        if i - last < gap:
            continue
        if rr[i] <= rr[i - 1] and rr[i] <= rr[i + 1]:
            idxs.append(i)
            last = i
    return idxs


def estimate_precession_from_peri_lrl(A_unwrapped, radius_series, steps_per_orbit_eff,
                                      use_from, use_to):
    """
    Robust estimator:
      1) find perihelion indices from radius series
      2) sample unwrapped LRL angle at those indices
      3) take successive differences dA
      4) remove baseline by subtracting nearest multiple of 2*pi:
            dA_small = dA - 2*pi * round(dA / (2*pi))
      5) average over the perihelion window [use_from, use_to)
    Returns advance in rad/orbit (small number, e.g., ~5e-7 for Mercury).
    """
    A = np.unwrap(np.asarray(A_unwrapped, dtype=float))
    rr = np.asarray(radius_series, dtype=float)
    idxs = _peri_indices_from_radius(rr, steps_per_orbit_eff)
    if len(idxs) < 2:
        return 0.0
    A_peri = np.unwrap(A[idxs])
    dA = np.diff(A_peri)
    if dA.size == 0:
        return 0.0

    two_pi = 2.0 * math.pi
    # modulo baseline to nearest multiple of 2*pi (handles +/- orientation)
    dA_small = dA - two_pi * np.round(dA / two_pi)

    i0 = max(0, int(use_from))
    i1 = min(dA_small.size, int(use_to))
    if i1 <= i0:
        i0, i1 = 0, dA_small.size

    dAwin = dA_small[i0:i1]
    if dAwin.size == 0:
        dAwin = dA_small

    return float(np.mean(dAwin))


# ---------------------------- Integration core ----------------------------

def integrate_orbit(mu, c, a, e, steps_per_orbit, periods, integrator, epsilon,
                    max_total_steps=0, store_stride=1):
    """
    Returns unwrapped LRL angle series, unwrapped theta series, radii, dt, T.
    Performance controls:
      max_total_steps > 0    -> clamp total steps to this cap
      store_stride >= 1      -> store every Nth sample (integrate all steps)
    """
    T = 2.0 * math.pi * math.sqrt(a ** 3 / mu)
    dt = T / float(steps_per_orbit)
    steps = int(max(1, periods * T / dt))

    if isinstance(max_total_steps, (int, float)) and int(max_total_steps) > 0:
        steps = min(steps, int(max_total_steps))

    s_stride = int(store_stride) if isinstance(store_stride, (int, float)) else 1
    if s_stride < 1:
        s_stride = 1

    r, v = periapsis_state(mu, a, e)

    def lrl_angle(rr, vv):
        h_z = rr[0] * vv[1] - rr[1] * vv[0]
        Ax = vv[1] * h_z
        Ay = -vv[0] * h_z
        rmag = math.hypot(rr[0], rr[1])
        if rmag > 0.0:
            Ax -= mu * rr[0] / rmag
            Ay -= mu * rr[1] / rmag
        return math.atan2(Ay, Ax)

    # Allocate only for stored samples
    n_store = (steps + s_stride - 1) // s_stride
    A_thetas = np.zeros(n_store, dtype=float)
    theta_series = np.zeros(n_store, dtype=float)
    radius_series = np.zeros(n_store, dtype=float)

    def a_fn(rr, vv):
        return accel_edg(mu, c, rr, vv, epsilon)

    stepper = get_stepper(integrator)

    k = 0
    for i in range(steps):
        if (i % s_stride) == 0:
            A_thetas[k] = lrl_angle(r, v)
            theta_series[k] = math.atan2(r[1], r[0])
            radius_series[k] = math.hypot(r[0], r[1])
            k += 1
        r, v = stepper(r, v, dt, a_fn)

    if k < n_store:
        A_thetas = A_thetas[:k]
        theta_series = theta_series[:k]
        radius_series = radius_series[:k]

    return np.unwrap(A_thetas), np.unwrap(theta_series), radius_series, dt, T


# ---------------------------- Plot and A0 fit ----------------------------

def plot_convergence_png(out_dir, planet, integrator, dts_arr, adv_arr, gr_target, A0, sfx=""):
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot(dts_arr, adv_arr, marker="o", label="model")
    ax.axhline(gr_target, linestyle="--", label="GR")
    ax.axhline(A0, linestyle=":", label="extrapolated A0")
    ax.set_xlabel("dt (s)")
    ax.set_ylabel("advance (rad/orbit)")
    ax.set_title("R6 convergence {} ({})".format(planet, integrator))
    ax.grid(True, alpha=0.4)
    ax.invert_xaxis()
    ax.legend(loc="best")
    png_path = os.path.join(out_dir, "R6_perihelion_convergence_{}_{}{}.png".format(planet, integrator, sfx))
    fig.tight_layout()
    print("[R6] Writing PNG:", png_path)
    fig.savefig(png_path, dpi=160)
    plt.close(fig)


def fit_A0_and_plot(out_dir, planet, integrator, df, sfx=""):
    dts_arr = df["dt"].to_numpy(dtype=float)
    adv_arr = df["advance_rad_per_orbit"].to_numpy(dtype=float)
    gr = float(df["gr_rad_per_orbit"].iloc[0]) if "gr_rad_per_orbit" in df.columns else float("nan")

    # polynomial order for dt^p on the abscissa
    if integrator == "vv":
        p_order = 2
    elif integrator in ("yoshida4", "rk4"):
        p_order = 4
    else:
        p_order = 2

    x = dts_arr ** float(p_order)
    A0 = _stable_fit_intercept(x, adv_arr) if x.size >= 2 else float(adv_arr[-1])
    rel_err = abs(A0 - gr) / gr if gr != 0.0 else float("nan")
    plot_convergence_png(out_dir, planet, integrator, dts_arr, adv_arr, gr, A0, sfx=sfx)
    return float(A0), float(rel_err), float(gr), int(p_order), float(np.min(dts_arr)), float(np.max(dts_arr))


# ---------------------------- Single-job runner ----------------------------

def run_single_job(conf_path, planet, integrator):
    with open(conf_path, "r") as f:
        cfg = yaml.safe_load(f)

    out_dir = cfg["paths"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    sfx = _suffix_from_cfg(cfg)

    r6 = cfg["r6"]

    overrides = r6.get("steps_per_orbit_overrides", {})
    if planet in overrides:
        steps_src = overrides[planet]
    else:
        steps_src = r6.get("steps_per_orbit_list", [96000, 192000, 384000])
    steps_list = [int(x) for x in steps_src]
    print("[R6] Steps per orbit for {}: {}".format(planet, steps_list))

    periods = int(r6.get("periods", 120))
    epsilon = float(r6.get("epsilon", 1.0))
    use_from = int(r6.get("use_perihelia_from", 10))
    use_to = int(r6.get("use_perihelia_to", 110))

    max_total_steps = int(r6.get("max_total_steps", 0))
    store_stride = int(r6.get("store_stride", 1))
    if max_total_steps or store_stride != 1:
        print("[R6] Performance: max_total_steps={}, store_stride={}".format(max_total_steps, store_stride))

    consts = r6.get("constants", {})
    G = float(consts.get("G", 6.67430e-11))
    M_sun = float(consts.get("M_sun", 1.98847e30))
    c = float(consts.get("c", 2.99792458e8))
    mu = G * M_sun

    planets = r6.get("constants_planets", {})
    if planet not in planets:
        raise ValueError("Planet {} not found in r6.constants_planets".format(planet))
    a = float(planets[planet]["a"])
    e = float(planets[planet]["e"])

    gr_target = delta_phi_gr(G, M_sun, c, a, e)

    conv_csv = os.path.join(out_dir, "R6_perihelion_convergence_{}_{}{}.csv".format(planet, integrator, sfx))
    print("[R6] Per-integrator CSV path:", conv_csv)
    existing = pd.read_csv(conv_csv) if os.path.exists(conv_csv) else None

    advs, dts = [], []
    job_start = time.time()

    for sps in steps_list:
        if existing is not None and (existing["steps_per_orbit"] == int(sps)).any():
            row = existing[existing["steps_per_orbit"] == int(sps)].iloc[0]
            advs.append(float(row["advance_rad_per_orbit"]))
            dts.append(float(row["dt"]))
            continue

        A_thetas, theta_series, radius_series, dt, T = integrate_orbit(
            mu=mu, c=c, a=a, e=e,
            steps_per_orbit=int(sps), periods=periods,
            integrator=integrator, epsilon=epsilon,
            max_total_steps=max_total_steps, store_stride=store_stride
        )

        sps_eff = max(1, int(int(sps) // max(1, store_stride)))

        # Legacy diagnostics
        adv_lrl = measure_precession_from_A(A_thetas, sps_eff, use_from, use_to)
        adv_sub = detect_perihelion_advance_subsample(theta_series, radius_series, sps_eff)
        adv_fit = estimate_precession_linear_fit(A_thetas, sps_eff, use_from, use_to)

        # Robust estimator (the one we intend to use)
        adv_periA = estimate_precession_from_peri_lrl(
            A_thetas, radius_series, sps_eff, use_from, use_to
        )

        # Choose final advance:
        # Prefer the robust perihelion-anchored value if finite and small in magnitude.
        final_adv = None
        if np.isfinite(adv_periA) and abs(adv_periA) < 0.1:
            final_adv = float(adv_periA)
            picked = "periA"
        else:
            # fallback to the smallest-magnitude among legacy values that are < 0.1 rad
            candidates = []
            if np.isfinite(adv_sub) and abs(adv_sub) < 0.1:
                candidates.append(("sub", float(adv_sub)))
            if np.isfinite(adv_lrl) and abs(adv_lrl) < 0.1:
                candidates.append(("lrl", float(adv_lrl)))
            if np.isfinite(adv_fit) and abs(adv_fit) < 0.1:
                candidates.append(("fit", float(adv_fit)))
            if candidates:
                picked, final_adv = sorted(candidates, key=lambda kv: abs(kv[1]))[0]
            else:
                picked, final_adv = ("fallback0", 0.0)

        print("[R6] {} {} sps={} (eff {}) dt={:.3f} advance={:.3e}  [periA={:.3e}, sub={:.3e}, lrl={:.3e}, fit={:.3e}] pick={}"
              .format(planet, integrator, int(sps), sps_eff, dt, final_adv,
                      float(adv_periA), float(adv_sub), float(adv_lrl), float(adv_fit), picked))

        advs.append(final_adv)
        dts.append(dt)

        row = {
            "planet": planet,
            "integrator": integrator,
            "steps_per_orbit": int(sps),
            "dt": float(dt),
            "advance_rad_per_orbit": float(final_adv),
            "advance_method": picked,
            "advance_sub_rad_per_orbit": float(adv_sub),
            "advance_lrl_rad_per_orbit": float(adv_lrl),
            "advance_fit_rad_per_orbit": float(adv_fit),
            "gr_rad_per_orbit": float(gr_target),
            "epsilon_used": float(epsilon),
            "periods": int(periods),
        }
        if existing is None:
            existing = pd.DataFrame([row])
        else:
            existing = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        existing.to_csv(conv_csv, index=False)

    df_now = pd.read_csv(conv_csv)
    A0, rel_err, _, p_order, _, _ = fit_A0_and_plot(out_dir, planet, integrator, df_now, sfx=sfx)

    elapsed = time.time() - job_start
    print("[R6] Finished {} with {} in {:.1f} seconds (A0={:.3e}, rel_err={:.3e}, p_order={})"
          .format(planet, integrator, elapsed, A0, rel_err, p_order))


# ---------------------------- Summary and replot helpers ----------------------------

def aggregate_summary_from_csvs(out_dir, planets, integrators, sfx=""):
    """
    Build a single summary table. Tries suffixed files first, then falls back.
    Summary file itself is always suffixed when sfx is non-empty.
    """
    rows = []
    for p in planets:
        for integ in integrators:
            path = os.path.join(out_dir, "R6_perihelion_convergence_{}_{}{}.csv".format(p, integ, sfx))
            if not os.path.exists(path):
                fallback = os.path.join(out_dir, "R6_perihelion_convergence_{}_{}.csv".format(p, integ))
                if os.path.exists(fallback):
                    path = fallback
                else:
                    print("[R6] Warning: missing", path)
                    continue

            df = pd.read_csv(path)
            A0, rel_err, gr, p_order, min_dt, max_dt = fit_A0_and_plot(out_dir, p, integ, df, sfx=sfx)
            eps_used = float(df.get("epsilon_used", pd.Series([np.nan])).iloc[0])
            periods = int(df.get("periods", pd.Series([np.nan])).iloc[0])
            rows.append({
                "planet": p,
                "integrator": integ,
                "epsilon_used": float(eps_used),
                "gr_rad_per_orbit": float(gr),
                "A0_extrapolated": float(A0),
                "relative_error": float(rel_err),
                "min_dt": float(min_dt),
                "max_dt": float(max_dt),
                "periods": int(periods),
                "p_order": int(p_order),
            })

    summary = pd.DataFrame(rows)
    summary_name = "R6_perihelion_summary{}.csv".format(sfx)
    summary_path = os.path.join(out_dir, summary_name)
    print("[R6] Writing summary:", summary_path)
    summary.to_csv(summary_path, index=False)
    return summary_path


def replot_all_from_csvs(conf_path):
    with open(conf_path, "r") as f:
        cfg = yaml.safe_load(f)
    out_dir = cfg["paths"]["out_dir"]
    r6 = cfg["r6"]
    planets = list(r6.get("planets", ["Mercury", "Venus", "Earth"]))
    integrators = list(r6.get("integrators", ["vv", "rk4"]))
    sfx = _suffix_from_cfg(cfg)
    return aggregate_summary_from_csvs(out_dir, planets, integrators, sfx=sfx)


def rebuild_summary_and_plots(conf_path):
    with open(conf_path, "r") as f:
        cfg = yaml.safe_load(f)

    out_dir = cfg["paths"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    r6 = cfg["r6"]
    planets = list(r6.get("planets", ["Mercury", "Venus", "Earth"]))
    integrators = list(r6.get("integrators", ["vv", "rk4"]))
    sfx = _suffix_from_cfg(cfg)

    rows = []
    for p in planets:
        for integ in integrators:
            conv_csv = os.path.join(out_dir, "R6_perihelion_convergence_{}_{}{}.csv".format(p, integ, sfx))
            if not os.path.exists(conv_csv):
                conv_csv_nosfx = os.path.join(out_dir, "R6_perihelion_convergence_{}_{}.csv".format(p, integ))
                if os.path.exists(conv_csv_nosfx):
                    conv_csv = conv_csv_nosfx
                else:
                    print("[R6] Warning: missing", conv_csv)
                    continue

            df = pd.read_csv(conv_csv)
            dts_arr = df["dt"].to_numpy(dtype=float)
            adv_arr = df["advance_rad_per_orbit"].to_numpy(dtype=float)
            gr = float(df["gr_rad_per_orbit"].iloc[0]) if "gr_rad_per_orbit" in df.columns else float("nan")

            if integ == "vv":
                order = 2
            elif integ in ("yoshida4", "rk4"):
                order = 4
            else:
                order = 2

            x = dts_arr ** float(order)
            A0 = _stable_fit_intercept(x, adv_arr) if x.size >= 2 else float(adv_arr[-1])
            rel_err = abs(A0 - gr) / gr if gr != 0.0 else float("nan")

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(dts_arr, adv_arr, marker="o", label="model")
            ax.axhline(gr, linestyle="--", label="GR")
            ax.axhline(A0, linestyle=":", label="extrapolated A0")
            ax.set_xlabel("dt (s)")
            ax.set_ylabel("advance (rad/orbit)")
            ax.set_title("R6 convergence {} ({})".format(p, integ))
            ax.grid(True, alpha=0.4)
            ax.invert_xaxis()
            ax.legend(loc="best")
            png_path = os.path.join(out_dir, "R6_perihelion_convergence_{}_{}{}.png".format(p, integ, sfx))
            fig.tight_layout()
            print("[R6] Rewriting PNG:", png_path)
            fig.savefig(png_path, dpi=160)
            plt.close(fig)

            rows.append({
                "planet": p,
                "integrator": integ,
                "epsilon_used": float(df.get("epsilon_used", pd.Series([np.nan])).iloc[0]),
                "gr_rad_per_orbit": float(gr),
                "A0_extrapolated": float(A0),
                "relative_error": float(rel_err),
                "min_dt": float(np.min(dts_arr)),
                "max_dt": float(np.max(dts_arr)),
                "periods": int(df.get("periods", pd.Series([np.nan])).iloc[0]),
                "p_order": int(order),
            })

    summary_name = "R6_perihelion_summary{}.csv".format(sfx)
    summary_path = os.path.join(out_dir, summary_name)
    print("[R6] Rewriting summary:", summary_path)
    pd.DataFrame(rows).to_csv(summary_path, index=False)


# ---------------------------- Orchestrator ----------------------------

def main(argv=None):
    if argv is None:
        argv = []
    ap = argparse.ArgumentParser()
    default_conf = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
    ap.add_argument("--conf", default=default_conf, help="Path to YAML config.")
    ap.add_argument("--planet", default=None, help="If set, run only this planet.")
    ap.add_argument("--integrator", default=None, help="If set, run only this integrator (vv, rk4, yoshida4).")
    ap.add_argument("--no_spawn", action="store_true", help="Run in-process without spawning children.")
    ap.add_argument("--replot", action="store_true", help="Rebuild plots and summary from existing CSVs only.")
    args = ap.parse_args(argv)

    if args.replot:
        rebuild_summary_and_plots(args.conf)
        return

    with open(args.conf, "r") as f:
        cfg = yaml.safe_load(f)

    out_dir = cfg["paths"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    r6 = cfg["r6"]
    planets = list(r6.get("planets", ["Mercury", "Venus", "Earth"]))
    integrators = list(r6.get("integrators", ["vv", "rk4"]))

    # Single-process, filtered path
    if args.planet or args.integrator or args.no_spawn:
        if args.planet:
            planets = [p for p in planets if p == args.planet]
        if args.integrator:
            integrators = [i for i in integrators if i == args.integrator]
        t0 = time.time()
        for p in planets:
            for integ in integrators:
                run_single_job(args.conf, p, integ)
        sfx = _suffix_from_cfg(cfg)
        aggregate_summary_from_csvs(out_dir, planets, integrators, sfx=sfx)
        print("[R6] All selected jobs completed in {:.1f} seconds".format(time.time() - t0))
        return

    # Fan-out (avoid on small laptops; R12 often calls without no_spawn)
    procs = []
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")

    for p in planets:
        for integ in integrators:
            cmd = [
                sys.executable, os.path.abspath(__file__),
                "--conf", args.conf, "--planet", p, "--integrator", integ, "--no_spawn"
            ]
            procs.append(subprocess.Popen(cmd, env=env))

    rc = 0
    for pr in procs:
        pr.wait()
        if pr.returncode != 0:
            rc = 1

    sfx = _suffix_from_cfg(cfg)
    aggregate_summary_from_csvs(out_dir, planets, integrators, sfx=sfx)
    sys.exit(rc)


def run(conf_path=None, **kwargs):
    """
    Import-friendly entry point. Swallows SystemExit so callers do not die.
    """
    argv = []
    if conf_path:
        argv += ["--conf", str(conf_path)]
    if kwargs.get("planet"):
        argv += ["--planet", kwargs["planet"]]
    if kwargs.get("integrator"):
        argv += ["--integrator", kwargs["integrator"]]
    if kwargs.get("no_spawn"):
        argv += ["--no_spawn"]
    if kwargs.get("replot"):
        argv += ["--replot"]
    try:
        return main(argv)
    except SystemExit as e:
        return int(getattr(e, "code", 0) or 0)


if __name__ == "__main__":
    main([])
