# =============================================================================
# Solar System Orbits (Entanglement-Driven Variant)
# Author: Kenneth Young, PhD
# Last updated: 2025-08-19
#
# Summary
# -------
# 3D solar system animation with mildly elliptical, inclined orbits integrated
# via Velocity Verlet. The central pull k(t) can be modulated by your quantum
# outputs (entropy.csv, hawking_radiation.csv) and optionally by a 3x3 curvature
# lattice (curvature_lattice.csv). Includes safety guardrails to avoid close
# approaches and CLI controls for Earth and Mars distances.
#
# Outputs
# -------
# - spacetime_outputs/animated_quantum_earth_orbit.gif
# - spacetime_outputs/animated_quantum_earth_orbit.mp4   (if ffmpeg available)
# - spacetime_outputs/earth_entropy.png
# - spacetime_outputs/close_Earth_Venus.png
# - spacetime_outputs/close_Earth_Mars.png
# =============================================================================

import os
import argparse
import matplotlib
matplotlib.use("Agg")  # render off-screen; avoid GUI overhead
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import pandas as pd

# -------------------------
# CLI
# -------------------------
parser = argparse.ArgumentParser(description="Entanglement-driven solar system animation.")
parser.add_argument("--earth_a", type=float, default=4.0, help="Earth semi-major axis (AU). Default 4.0")
parser.add_argument("--mars_a",  type=float, default=7.0, help="Mars semi-major axis (AU). Default 7.0")
parser.add_argument("--use_entanglement", action="store_true", help="Enable entanglement modulation of k(t).")
parser.add_argument("--use_spatial_curvature", action="store_true", help="Enable 3x3 curvature lattice local modulation.")
parser.add_argument("--trail_len", type=int, default=120, help="Frames to keep visible as comet-like trail.")
parser.add_argument("--no_trails", action="store_true", help="Disable trails (draw full orbit progress).")
parser.add_argument("--n_steps", type=int, default=1000, help="Integration steps and animation frames. Default 1000")
parser.add_argument("--dt", type=float, default=0.01, help="Integrator time step. Default 0.01")
parser.add_argument("--safe_min_ev", type=float, default=0.8, help="Min Earth-Venus separation in AU to pass guardrail.")
parser.add_argument("--safe_min_em", type=float, default=1.2, help="Min Earth-Mars separation in AU to pass guardrail.")
parser.add_argument("--no_assert_guardrail", action="store_true", help="Do not raise on guardrail failure, only warn.")
args, _ = parser.parse_known_args()

# -------------------------
# Config
# -------------------------
OUTPUT_DIR = "spacetime_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DT = float(args.dt)
N_STEPS = int(args.n_steps)
SAVE_GIF_PATH = os.path.join(OUTPUT_DIR, "animated_quantum_earth_orbit.gif")
SAVE_MP4_PATH = os.path.join(OUTPUT_DIR, "animated_quantum_earth_orbit.mp4")
SAVE_ENTROPY_PNG = os.path.join(OUTPUT_DIR, "earth_entropy.png")

USE_ENTANGLEMENT_DRIVE = bool(args.use_entanglement)        # off by default
USE_SPATIAL_CURVATURE_MOD = bool(args.use_spatial_curvature)
USE_TRAILS = not bool(args.no_trails)
TRAIL_LEN = int(args.trail_len)

USE_ENTANGLEMENT_DRIVE = True
USE_SPATIAL_CURVATURE_MOD = True

K0 = 100.0                          # base pull constant
ALPHA_ENTR_RATE = 0.02              # weight on normalized dEntropy/dt
ALPHA_HAWK     = 0.01               # weight on normalized Hawking proxy
EPS_CAP = 0.10                      # cap total fractional modulation
BETA_LOCAL = 0.05                   # local strength from curvature cell (0..1 -> up to 5 percent)
RMAX = 110.0                        # sets outer plot span and pos->cell mapping

SAFE_MIN_EV = float(args.safe_min_ev)
SAFE_MIN_EM = float(args.safe_min_em)
RAISE_ON_FAIL = not bool(args.no_assert_guardrail)

# -------------------------
# Planet parameters
# -------------------------
sun_pos = np.array([0.0, 0.0, 0.0])

a_dict = {
    'Mercury': 1,
    'Venus':   2,
    'Earth':   args.earth_a,  # CLI
    'Mars':    args.mars_a,   # CLI
    'Jupiter': 17, 'Saturn': 32, 'Uranus': 64, 'Neptune': 100
}

colors = {
    'Mercury': 'gray', 'Venus': 'orange', 'Earth': 'blue', 'Mars': 'red',
    'Jupiter': 'brown', 'Saturn': 'gold', 'Uranus': 'cyan', 'Neptune': 'darkblue'
}

inner_planets = ['Mercury', 'Venus', 'Earth', 'Mars']
outer_planets = ['Jupiter', 'Saturn', 'Uranus', 'Neptune']

eccentricities = {
    'Mercury': 0.20,
    'Venus':   0.006,   # lowered
    'Earth':   0.010,   # lowered
    'Mars':    0.080,   # lowered
    'Jupiter': 0.048,
    'Saturn':  0.056,
    'Uranus':  0.047,
    'Neptune': 0.009
}
inclinations_deg = {
    'Mercury': 7.0, 'Venus': 3.4, 'Earth': 0.0, 'Mars': 1.85,
    'Jupiter': 1.3, 'Saturn': 2.5, 'Uranus': 0.8, 'Neptune': 1.8
}

# safe separated initial phases
SAFE_PHASES = True
phase_offset = {
    'Mercury': np.deg2rad(0),
    'Venus':   np.deg2rad(135),
    'Earth':   np.deg2rad(270),
    'Mars':    np.deg2rad(45),
    'Jupiter': np.deg2rad(120),
    'Saturn':  np.deg2rad(210),
    'Uranus':  np.deg2rad(300),
    'Neptune': np.deg2rad(30),
}

# -------------------------
# Entanglement data loading
# -------------------------
def _norm01(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or hi - lo == 0.0:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)

def load_series(path, value_col=1):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        vals = df.iloc[:, value_col].to_numpy(dtype=float)
        return vals
    except Exception:
        return None

def resample_to_steps(arr, steps_plus_one):
    if arr is None or arr.size == 0:
        return None
    idx = np.round(np.linspace(0, arr.size - 1, steps_plus_one)).astype(int)
    return arr[idx]

E_src = load_series(os.path.join(OUTPUT_DIR, "entropy.csv"))
H_src = load_series(os.path.join(OUTPUT_DIR, "hawking_radiation.csv"))
dE_src = None
if E_src is not None:
    dE_src = np.zeros_like(E_src); dE_src[1:] = np.diff(E_src)

E_series  = resample_to_steps(E_src, N_STEPS + 1) if USE_ENTANGLEMENT_DRIVE else None
dE_series = resample_to_steps(dE_src, N_STEPS + 1) if USE_ENTANGLEMENT_DRIVE else None
H_series  = resample_to_steps(H_src, N_STEPS + 1) if USE_ENTANGLEMENT_DRIVE else None

curv_lat_df = None
if USE_SPATIAL_CURVATURE_MOD:
    path_curv = os.path.join(OUTPUT_DIR, "curvature_lattice.csv")
    if os.path.exists(path_curv):
        try:
            curv_lat_df = pd.read_csv(path_curv, index_col=0)
        except Exception:
            curv_lat_df = None

def curv_grid_for_step(step):
    if curv_lat_df is None:
        return np.zeros((3, 3), dtype=float)
    try:
        ncols = curv_lat_df.shape[1]
        col_idx = int(round((ncols - 1) * step / N_STEPS))
        vals = curv_lat_df.iloc[:9, col_idx].values.astype(float).reshape(3, 3)
        lo, hi = np.nanmin(vals), np.nanmax(vals)
        if not np.isfinite(lo) or hi - lo == 0.0:
            return np.zeros_like(vals)
        return (vals - lo) / (hi - lo)
    except Exception:
        return np.zeros((3, 3), dtype=float)

def k_eff_global(step):
    if not USE_ENTANGLEMENT_DRIVE or dE_series is None or H_series is None:
        return K0
    dE_n = _norm01(dE_series)
    H_n  = _norm01(H_series)
    eps = ALPHA_ENTR_RATE * dE_n[step] + ALPHA_HAWK * H_n[step]
    eps = float(np.clip(eps, -EPS_CAP, EPS_CAP))
    return K0 * (1.0 + eps)

def pos_to_cell(x, y, rmax=RMAX):
    def to_idx(val):
        u = (val + rmax) / (2.0 * rmax)
        return int(np.clip(np.floor(u * 3.0), 0, 2))
    return to_idx(x), to_idx(y)

# -------------------------
# Orbital helpers
# -------------------------
def rotate_about_x(vec, deg):
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    R = np.array([[1, 0, 0],
                  [0, c, -s],
                  [0, s,  c]], dtype=float)
    return R @ vec

def initial_state_periapsis(a, e, inc_deg, phase):
    mu = K0  # base mu for initial conditions
    r_peri = a * (1.0 - e)
    v_peri = np.sqrt(mu * (1.0 + e) / (a * (1.0 - e)))
    cp, sp = np.cos(phase), np.sin(phase)
    pos2d = np.array([ r_peri*cp, r_peri*sp, 0.0 ])
    vel2d = np.array([-v_peri*sp, v_peri*cp, 0.0 ])
    pos3d = rotate_about_x(pos2d, inc_deg)
    vel3d = rotate_about_x(vel2d, inc_deg)
    return pos3d, vel3d

def accel_at(pos, k_param, local_mult=1.0):
    r_vec = pos - sun_pos
    dist2 = float(np.dot(r_vec, r_vec)) + 1e-9
    inv_r = 1.0 / np.sqrt(dist2)
    r_hat = r_vec * inv_r
    k_eff = k_param * local_mult
    return -k_eff * r_hat / dist2

# -------------------------
# Build initial conditions
# -------------------------
print("Starting solar system simulation...")
positions, velocities = {}, {}
for planet, a in a_dict.items():
    e   = float(eccentricities.get(planet, 0.02))
    inc = float(inclinations_deg.get(planet, 0.0))
    ph  = float(phase_offset[planet])
    p0, v0 = initial_state_periapsis(a, e, inc, ph)
    positions[planet] = p0
    velocities[planet] = v0


# --------------------------------------------------
# Fast entropy plot (Earth) 
# Key speedups:
#  - Non-interactive backend (Agg)
#  - Optional decimation for very long series
#  - No tight_layout(), no bbox_inches="tight"
#  - Antialiasing off, small linewidth
#  - Path simplification on
# --------------------------------------------------
def _decimate_xy(x, y, max_points=100_000):
    """
    Downsample (x,y) if longer than max_points using stride decimation.
    Keeps endpoints to preserve extents. O(1) memory.
    """
    n = len(x)
    if n <= max_points or max_points <= 0:
        return x, y
    step = max(1, n // max_points)
    # Keep first/last for exact bounds
    x_dec = np.concatenate([x[::step], x[-1:]])
    y_dec = np.concatenate([y[::step], y[-1:]])
    return x_dec, y_dec

def plot_entropy_fast(t, s, out_path, title="Entanglement Entropy (Earth)",
                      max_points=100_000, linewidth=1.0):
    # Ensure ndarray (float32 reduces memory/bandwidth for large series)
    t = np.asarray(t, dtype=np.float32)
    s = np.asarray(s, dtype=np.float32)

    # Optional decimation for very long series
    t_plot, s_plot = _decimate_xy(t, s, max_points=max_points)

    # Lightweight figure setup
    plt.rcParams["path.simplify"] = True
    plt.rcParams["path.simplify_threshold"] = 0.5  # larger => more simplification

    fig, ax = plt.subplots(figsize=(8, 4))
    line, = ax.plot(
        t_plot, s_plot,
        label="Entanglement Entropy (Earth)",
        antialiased=False,
        linewidth=linewidth,
    )
    # If exporting to vector formats with many points, consider rasterizing the line:
    # line.set_rasterized(True)

    ax.set_xlabel("Time (scaled units)")
    ax.set_ylabel("Entropy")
    ax.set_title(title)
    ax.legend(loc="best", frameon=False)
    ax.margins(x=0.01, y=0.05)  # cheap margins; avoids tight_layout()

    # Save without tight/bbox passes (these can be slow on big plots)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)



trajectories = {p: [positions[p].copy()] for p in a_dict.keys()}

# synthetic entropy time vector for the example plot
t = np.linspace(0.0, DT * N_STEPS, N_STEPS + 1)
peak_time = 0.5 * t[-1]
peak_entropy = 1.5
base_entropy = 0.58
entropy = {p: np.zeros_like(t) for p in a_dict.keys()}
for p in a_dict.keys():
    for i, ti in enumerate(t):
        if ti <= peak_time:
            e_val = base_entropy + (peak_entropy - base_entropy) * (ti / peak_time)**2
        else:
            e_val = peak_entropy - (peak_entropy - base_entropy) * ((ti - peak_time) / (t[-1] - peak_time))**2
        entropy[p][i] = e_val

# -------------------------------
# Integrate with Velocity Verlet
# -------------------------------
print("Integrating motion using Velocity Verlet...")
for step in range(N_STEPS):
    if step % 100 == 0:
        print(f"Step {step}/{N_STEPS}...")
    k_now  = k_eff_global(step)
    k_next = k_eff_global(step + 1 if step + 1 <= N_STEPS else step)
    grid3 = curv_grid_for_step(step) if USE_SPATIAL_CURVATURE_MOD else None

    for planet in a_dict.keys():
        pos = positions[planet]
        vel = velocities[planet]

        local_mult_now = 1.0
        if grid3 is not None:
            ix, iy = pos_to_cell(pos[0], pos[1], rmax=RMAX)
            local_mult_now = 1.0 + BETA_LOCAL * float(grid3[iy, ix])

        a_now = accel_at(pos, k_now, local_mult_now)
        pos_new = pos + vel * DT + 0.5 * a_now * (DT**2)

        local_mult_new = 1.0
        if grid3 is not None:
            ix2, iy2 = pos_to_cell(pos_new[0], pos_new[1], rmax=RMAX)
            local_mult_new = 1.0 + BETA_LOCAL * float(grid3[iy2, ix2])

        a_new = accel_at(pos_new, k_next, local_mult_new)
        vel_new = vel + 0.5 * (a_now + a_new) * DT

        positions[planet] = pos_new
        velocities[planet] = vel_new
        trajectories[planet].append(pos_new.copy())

trajectories = {p: np.array(traj) for p, traj in trajectories.items()}
print("Motion integration completed.")

# ---------------------------------
# Separation checks and snapshots
# ---------------------------------
def min_pair_sep(traj_a, traj_b):
    d = np.linalg.norm(traj_a - traj_b, axis=1)
    return float(d.min())

def save_closeup(name, traj_a, traj_b):
    d = np.linalg.norm(traj_a - traj_b, axis=1)
    idx = int(np.argmin(d))
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.plot(traj_a[:, 0], traj_a[:, 1], label=name.split('_')[0])
    ax.plot(traj_b[:, 0], traj_b[:, 1], label=name.split('_')[1])
    ax.scatter([traj_a[idx, 0]], [traj_a[idx, 1]], s=36)
    ax.scatter([traj_b[idx, 0]], [traj_b[idx, 1]], s=36)
    ax.set_aspect('equal', 'box')
    ax.legend(); fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"close_{name}.png")
    fig.savefig(path, dpi=160); plt.close(fig)
    print(f"Saved {path}")

min_ev = min_pair_sep(trajectories['Earth'], trajectories['Venus'])
min_em = min_pair_sep(trajectories['Earth'], trajectories['Mars'])
print(f"Minimum Earth-Venus separation: {min_ev:.3f} AU")
print(f"Minimum Earth-Mars separation:  {min_em:.3f} AU")

save_closeup("Earth_Venus", trajectories['Earth'], trajectories['Venus'])
save_closeup("Earth_Mars",  trajectories['Earth'],  trajectories['Mars'])

if RAISE_ON_FAIL and (min_ev < SAFE_MIN_EV or min_em < SAFE_MIN_EM):
    raise RuntimeError(
        f"Separation too small: EV={min_ev:.3f} AU (min {SAFE_MIN_EV}), "
        f"EM={min_em:.3f} AU (min {SAFE_MIN_EM})"
    )
elif (min_ev < SAFE_MIN_EV or min_em < SAFE_MIN_EM):
    print("Warning: separation below guardrail thresholds.")

# -------------------------
# Animation setup
# -------------------------
print("Setting up the animation...")
fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(121, projection='3d')
ax1.set_xlim(-8, 8);   ax1.set_ylim(-8, 8);   ax1.set_zlim(-1.5, 1.5)
ax1.set_title("Inner Planets")
ax1.set_xlabel("X (AU)"); ax1.set_ylabel("Y (AU)"); ax1.set_zlabel("Z")
ax1.scatter([0], [0], [0], color='gold', s=150, label="Sun")

ax2 = fig.add_subplot(122, projection='3d')
ax2.set_xlim(-RMAX, RMAX); ax2.set_ylim(-RMAX, RMAX); ax2.set_zlim(-5, 5)
ax2.set_title("Outer Planets")
ax2.set_xlabel("X (AU)"); ax2.set_ylabel("Y (AU)"); ax2.set_zlabel("Z")
ax2.scatter([0], [0], [0], color='gold', s=150, label="Sun")

orbit_lines1, planet_dots1 = {}, {}
orbit_lines2, planet_dots2 = {}, {}
for planet in a_dict.keys():
    if planet in inner_planets:
        orbit_lines1[planet], = ax1.plot([], [], [], lw=2, color=colors[planet], label=planet)
        planet_dots1[planet], = ax1.plot([], [], [], 'o', color=colors[planet], markersize=5)
    else:
        orbit_lines2[planet], = ax2.plot([], [], [], lw=2, color=colors[planet], label=planet)
        planet_dots2[planet], = ax2.plot([], [], [], 'o', color=colors[planet], markersize=5)

ax1.legend(loc='upper right')
ax2.legend(loc='upper right')

def update(frame):
    for planet in a_dict.keys():
        traj = trajectories[planet]
        if USE_TRAILS:
            end = frame + 1
            start = max(0, end - TRAIL_LEN)
            seg = traj[start:end]
        else:
            seg = traj[:frame+1]
        x, y, z = seg[:, 0], seg[:, 1], seg[:, 2]
        if planet in inner_planets:
            orbit_lines1[planet].set_data(x, y)
            orbit_lines1[planet].set_3d_properties(z)
            planet_dots1[planet].set_data([x[-1]], [y[-1]])
            planet_dots1[planet].set_3d_properties([z[-1]])
        else:
            orbit_lines2[planet].set_data(x, y)
            orbit_lines2[planet].set_3d_properties(z)
            planet_dots2[planet].set_data([x[-1]], [y[-1]])
            planet_dots2[planet].set_3d_properties([z[-1]])
    return list(orbit_lines1.values()) + list(planet_dots1.values()) + list(orbit_lines2.values()) + list(planet_dots2.values())

print("Creating and saving the animation...")
ani = animation.FuncAnimation(fig, update, frames=N_STEPS, interval=40, blit=False)
ani.save(SAVE_GIF_PATH, writer="pillow", fps=24)
print(f"GIF saved to {SAVE_GIF_PATH}.")

try:
    from matplotlib.animation import FFMpegWriter
    writer = FFMpegWriter(fps=24, bitrate=3000)
    ani.save(SAVE_MP4_PATH, writer=writer)
    print(f"MP4 saved to {SAVE_MP4_PATH}.")
except Exception as e:
    print(f"MP4 export skipped (ffmpeg not available): {e}")

# Plot entanglement entropy  for Earth
print("Saving entanglement entropy plot for Earth...")
plot_entropy_fast(
    t=t,
    s=entropy["Earth"],
    out_path=SAVE_ENTROPY_PNG,
    title="Entanglement Entropy During Earth's Orbit (synthetic)",
    max_points=100_000,  # tune: lower for faster/smaller files, higher for more detail
    linewidth=1.0
)
print(f"Entropy plot saved to {SAVE_ENTROPY_PNG}.")

print("Simulation and visualization completed successfully!")
