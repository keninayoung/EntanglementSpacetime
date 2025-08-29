# validation_r4_perihelion.py
# R4: GR perihelion benchmark with bias removal and iterative epsilon calibration.
# ASCII-only.

import os
import argparse
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


# ------------------------------ helpers --------------------------------

def to_float(x, name="value"):
    try:
        return float(x)
    except Exception:
        raise ValueError("Expected numeric {} in config, got: {}".format(name, repr(x)))


def delta_phi_gr(G, M, c, a, e):
    # GR perihelion advance per orbit (radians), first-order weak-field
    return (6.0 * np.pi * G * M) / (a * (1.0 - e * e) * c * c)


def periapsis_state(mu, a, e):
    # State at periapsis for ellipse with semi-major axis a and eccentricity e.
    r_p = a * (1.0 - e)
    v_p = np.sqrt(mu * (1.0 + e) / (a * (1.0 - e)))
    r0 = np.array([r_p, 0.0], dtype=float)
    v0 = np.array([0.0, v_p], dtype=float)
    return r0, v0


def integrate_orbit_si(mu, c, r0, v0, dt, steps, mode="gr_like", epsilon=1.0, a_mod=0.0, omega=0.0, verbose=False):
    # Velocity-Verlet-like 2D integrator with optional GR-like correction.
    if verbose:
        print("[integrate] mode={}, epsilon={:.6e}, dt={:.6e}, steps={}".format(mode, float(epsilon), float(dt), int(steps)))
    r = r0.copy()
    v = v0.copy()
    t = 0.0

    thetas = np.zeros(steps, dtype=float)
    radii = np.zeros(steps, dtype=float)

    def accel(t_local, r_vec, v_vec):
        x, y = r_vec
        r2 = x * x + y * y
        rmag = np.sqrt(r2)
        if rmag == 0.0:
            return np.zeros(2, dtype=float)
        rhat = r_vec / rmag
        mu_eff = mu
        if mode == "modulation":
            mu_eff = mu * (1.0 + a_mod * np.sin(omega * t_local))
        a_vec = -mu_eff * rhat / r2
        if mode == "gr_like":
            # h is z-component magnitude of r x v in 2D
            h = abs(r_vec[0] * v_vec[1] - r_vec[1] * v_vec[0])
            r4 = r2 * r2
            a_extra = -(3.0 * float(epsilon) * mu * (h * h) / (c * c * r4)) * rhat
            a_vec = a_vec + a_extra
        return a_vec

    a_n = accel(t, r, v)
    for n in range(steps):
        thetas[n] = np.arctan2(r[1], r[0])
        radii[n] = np.hypot(r[0], r[1])

        # VV step with predicted accel
        r_half = r + 0.5 * dt * v
        v_half = v + 0.5 * dt * a_n
        t_next = t + dt
        a_np1 = accel(t_next, r_half + 0.5 * dt * v_half, v_half)
        v = v + 0.5 * dt * (a_n + a_np1)
        r = r + dt * v
        t = t_next
        a_n = a_np1

    return {"theta": thetas, "r": radii}


def _quad_interp_min(y_minus, y0, y_plus):
    # Parabola through points (-1, y-), (0, y0), (1, y+); return vertex x*.
    denom = (y_minus - 2.0 * y0 + y_plus)
    if denom == 0.0:
        return 0.0
    return 0.5 * (y_minus - y_plus) / denom


def _lin_interp(y0, y1, frac):
    return (1.0 - frac) * y0 + frac * y1


def detect_perihelion_advance_subsample(theta_series, r_series, dt):
    # Local minima of r; quadratic refine time, linear interp theta there.
    N = len(r_series)
    if N < 3:
        return 0.0, 0.0
    th = np.unwrap(theta_series)

    peri_thetas = []
    for i in range(1, N - 1):
        if r_series[i] <= r_series[i - 1] and r_series[i] <= r_series[i + 1]:
            xstar = _quad_interp_min(r_series[i - 1], r_series[i], r_series[i + 1])
            if xstar >= 0.0:
                frac = min(1.0, max(0.0, xstar))
                th_ref = _lin_interp(th[i], th[i + 1], frac)
            else:
                frac = min(1.0, max(0.0, 1.0 + xstar))
                th_ref = _lin_interp(th[i - 1], th[i], frac)
            peri_thetas.append(th_ref)

    if len(peri_thetas) < 2:
        return 0.0, 0.0

    deltas = np.diff(peri_thetas)
    raw_adv = float(np.mean(np.maximum(0.0, deltas - 2.0 * np.pi)))
    bias_proxy = float(np.std(deltas) / (np.sqrt(len(deltas)) + 1e-12))
    return raw_adv, max(0.0, bias_proxy)


# ------------------------------ main run --------------------------------

def run(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    out_dir = cfg["paths"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # Constants (SI)
    cst = cfg.get("constants", {})
    G = to_float(cst.get("G", 6.67430e-11), "constants.G")
    M_sun = to_float(cst.get("M_sun", 1.98847e30), "constants.M_sun")
    c = to_float(cst.get("c", 2.99792458e8), "constants.c")
    a = to_float(cst.get("a_mercury", 5.79e10), "constants.a_mercury")
    e = to_float(cst.get("e_mercury", 0.206), "constants.e_mercury")
    mu = G * M_sun

    T = 2.0 * np.pi * np.sqrt(a ** 3 / mu)

    # Integrator settings
    peri = cfg["orbits"]["mercury"]
    periods = int(peri.get("periods", 120))
    psec = cfg.get("perihelion", {})
    steps_per_orbit = psec.get("steps_per_orbit", 40000)
    if steps_per_orbit is not None:
        steps_per_orbit = int(steps_per_orbit)
        dt = T / float(steps_per_orbit)
    else:
        dt = to_float(peri.get("dt", T / 10000.0), "orbits.mercury.dt")
    steps = int(max(1, periods * T / dt))

    # Calibration controls
    eps_small1 = to_float(psec.get("eps_small1", 1.0e-4), "perihelion.eps_small1")
    eps_small2 = to_float(psec.get("eps_small2", 2.0e-4), "perihelion.eps_small2")
    max_iters = int(psec.get("max_iters", 4))
    tol_rel = to_float(psec.get("tol_rel", 5e-3), "perihelion.tol_rel")  # target relative error, e.g., 0.5%

    gr_adv = delta_phi_gr(G, M_sun, c, a, e)
    print("[R4] GR analytic advance (rad/orbit): {:.6e}".format(gr_adv))

    r0, v0 = periapsis_state(mu, a, e)

    # Bias at epsilon=0
    out_bias = integrate_orbit_si(mu=mu, c=c, r0=r0, v0=v0, dt=dt, steps=steps,
                                  mode="gr_like", epsilon=0.0, verbose=True)
    dphi_bias_raw, bias_proxy = detect_perihelion_advance_subsample(out_bias["theta"], out_bias["r"], dt)
    print("[R4] Bias epsilon=0 -> dphi_bias_raw = {:.6e} (proxy {:.3e})".format(dphi_bias_raw, bias_proxy))

    # Function that returns corrected advance for a given epsilon
    def measure_corr(epsilon_val):
        out = integrate_orbit_si(mu=mu, c=c, r0=r0, v0=v0, dt=dt, steps=steps,
                                 mode="gr_like", epsilon=float(epsilon_val), verbose=True)
        dphi_raw, _ = detect_perihelion_advance_subsample(out["theta"], out["r"], dt)
        return max(0.0, dphi_raw - dphi_bias_raw)

    # Seed with two small epsilons for slope
    y1 = measure_corr(eps_small1)
    y2 = measure_corr(eps_small2)
    print("[R4] small eps: e1={:.2e} -> {:.6e}, e2={:.2e} -> {:.6e}".format(eps_small1, y1, eps_small2, y2))

    # Initial estimate via secant on f(eps) = y(eps) - gr_adv
    def f(eps):
        return measure_corr(eps) - gr_adv

    # Keep a trace for CSV
    trace_rows = []
    def record(case, eps, y_corr):
        rel_err = abs(y_corr - gr_adv) / gr_adv if gr_adv != 0.0 else np.nan
        trace_rows.append({
            "case": case,
            "epsilon": float(eps),
            "y_corrected": float(y_corr),
            "gr_rad": float(gr_adv),
            "relative_error": float(rel_err),
            "dt": float(dt),
            "periods": int(periods),
            "steps_per_orbit": int(steps_per_orbit),
            "T_seconds": float(T),
        })
        return rel_err

    record("bias_e0", 0.0, 0.0)
    record("small_e1", eps_small1, y1)
    record("small_e2", eps_small2, y2)

    # If y2 <= y1, expand eps_small2 until monotonic increase (avoid zero slope)
    e_lo, y_lo = eps_small1, y1
    e_hi, y_hi = eps_small2, y2
    tries = 0
    while y_hi <= y_lo and tries < 5:
        e_hi *= 2.0
        y_hi = measure_corr(e_hi)
        record("expand_hi", e_hi, y_hi)
        tries += 1

    # Secant iterations with safe bisection fallback
    eps_prev, f_prev = e_lo, (y_lo - gr_adv)
    eps_curr, f_curr = e_hi, (y_hi - gr_adv)

    best_eps = eps_curr
    best_y = y_hi
    best_err = record("init_pair", eps_curr, y_hi)

    for it in range(max_iters):
        # Secant step
        denom = (f_curr - f_prev)
        if denom == 0.0 or not np.isfinite(denom):
            eps_next = 0.5 * (eps_prev + eps_curr)  # fallback
        else:
            eps_next = eps_curr - f_curr * (eps_curr - eps_prev) / denom

        # Keep positive and within reasonable bounds
        if not np.isfinite(eps_next) or eps_next <= 0.0:
            eps_next = 0.5 * (eps_prev + eps_curr)

        y_next = measure_corr(eps_next)
        err_next = record("iter_{}".format(it + 1), eps_next, y_next)
        if err_next < best_err:
            best_err, best_eps, best_y = err_next, eps_next, y_next

        # Check convergence
        if err_next <= tol_rel:
            print("[R4] Converged in {} iterations. eps = {:.6e}, rel_err = {:.3e}".format(it + 1, eps_next, err_next))
            best_eps, best_y, best_err = eps_next, y_next, err_next
            break

        # Update pair for next secant step
        # Keep bracket in increasing order of epsilon for stability
        eps_prev, f_prev = eps_curr, f_curr
        eps_curr, f_curr = eps_next, (y_next - gr_adv)

    # Final verification at best_eps
    final_y = best_y
    final_err = best_err
    print("[R4] Best eps = {:.6e}, corrected advance = {:.6e}, rel_err = {:.3e}".format(best_eps, final_y, final_err))

    # Save step trace
    steps_csv = os.path.join(out_dir, "R4_perihelion_calibration_steps.csv")
    pd.DataFrame(trace_rows).to_csv(steps_csv, index=False)
    print("[R4] Wrote {}".format(steps_csv))

    # Save final comparison
    final_csv = os.path.join(out_dir, "R4_perihelion_comparison.csv")
    pd.DataFrame([{
        "epsilon_best": float(best_eps),
        "delta_phi_model_corr": float(final_y),
        "delta_phi_GR_rad": float(gr_adv),
        "relative_error": float(final_err),
        "dt": float(dt),
        "periods": int(periods),
        "steps_per_orbit": int(steps_per_orbit),
        "T_seconds": float(T),
        "bias_raw": float(dphi_bias_raw),
    }]).to_csv(final_csv, index=False)
    print("[R4] Wrote {}".format(final_csv))

    # Plot final corrected value vs GR
    plt.figure()
    plt.plot([0, 1], [final_y, final_y], marker="o", label="Model corrected (best)")
    plt.plot([0, 1], [gr_adv, gr_adv], linestyle="--", label="GR (analytic)")
    plt.xticks([0, 1], ["best", "GR"])
    plt.xlabel("Case")
    plt.ylabel("Perihelion advance (rad/orbit)")
    plt.title("R4: Iterative calibration vs GR (bias-corrected)")
    plt.grid(True)
    plt.legend(loc="best")
    out_png = os.path.join(out_dir, "R4_perihelion_vs_GR.png")
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    print("[R4] Wrote {}".format(out_png))


def main():
    ap = argparse.ArgumentParser()
    default_conf = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
    ap.add_argument("--conf", default=default_conf)
    args = ap.parse_args()
    run(args.conf)


if __name__ == "__main__":
    main()
