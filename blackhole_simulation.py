# =============================================================================
# Black Hole Entanglement Animation (Smooth Monotonic Growth)
# Author: Kenneth Young, PhD
# Last updated: 2025-08-19
#
# What this does
# --------------
# - Reads quantum outputs produced by your PEPS pipeline:
#     spacetime_outputs/entropy.csv            columns: Step, Entropy
#     spacetime_outputs/hawking_radiation.csv  columns: Step, MI Across Horizon
#     spacetime_outputs/curvature_lattice.csv  optional, 3x3 values per step
#
# - Builds a star field with heterogeneous orbits and sizes.
# - Uses your signals to control:
#     * Global pull for inward drift and spin-up (Hawking MI and positive dEntropy/dt).
#     * Optional local sector effects from the 3x3 curvature lattice.
#     * Tiny swirl bias from the curvature gradient.
#
# - Draws the black hole as a filled disk whose radius equals the Schwarzschild
#   radius mapped into plot units:
#       r_s = 2 * G * M / c^2
#   The mapping from meters to plot units is computed once for a reference mass.
#   The effective radius then grows smoothly and monotonically using a smoothed
#   cumulative driver derived from your signals. It does not shrink.
#
# Output
# ------
# - spacetime_outputs/black_hole_entanglement_2d.gif
#
# Notes
# -----
# - ASCII only (no unicode).
# - Works even if curvature_lattice.csv is missing.
# - Tune behavior in the CONFIG section.
# =============================================================================

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.lines import Line2D
from PIL import Image

# =============================================================================
# CONFIG
# =============================================================================

# Timing and animation
TIME_STEPS = 20           # number of discrete quantum steps to key off
TOTAL_FRAMES = 100        # total animation frames
DT = 0.1                  # used only for the title display
FRAME_INTERVAL_MS = 100   # milliseconds per frame for GIF writer
FPS = 10                  # frames per second in GIF

# Data locations
OUTPUT_DIR = "spacetime_outputs"
CSV_FILES = {
    "hawking":  os.path.join(OUTPUT_DIR, "hawking_radiation.csv"),
    "entropy":  os.path.join(OUTPUT_DIR, "entropy.csv"),
    "curv_lat": os.path.join(OUTPUT_DIR, "curvature_lattice.csv"),  # optional
}
GIF_PATH = os.path.join(OUTPUT_DIR, "black_hole_entanglement_2d.gif")

# Plot window and optional background
XLIM = (-0.7, 0.7)
YLIM = (-0.7, 0.7)
BACKGROUND_IMG_PATH = os.path.join("images", "milkyway.jpg")  # optional

# Star field configuration
N_STARS = 24
R_MIN, R_MAX = 0.20, 0.62
ECC_MEAN, ECC_STD = 0.15, 0.07
ANG_VEL_RANGE = (0.035, 0.075)
DRAG_RANGE = (0.004, 0.010)
COUPLING_LOGN_MEAN = 0.0
COUPLING_LOGN_SIGMA = 0.35
STAR_SIZE_MIN, STAR_SIZE_MAX = 95, 205

# Quantum-driven dynamics
USE_SPATIAL_CURVATURE_MOD = True  # set False to ignore 3x3 modulation
BETA_LOCAL = 0.5                  # local curvature strength in [0,1]
SWIRL_GAIN_SCALE = 0.03           # scales swirl by global pull

# Pull strength weights for stellar motion (non-monotonic OK)
PULL_W_ENTR_POS = 0.6             # positive part of normalized dEntropy/dt
PULL_W_HAWK = 0.5                 # normalized Hawking MI

# Schwarzschild mapping and monotonic growth
# Physical constants (SI)
G_SI = 6.67430e-11                # m^3 kg^-1 s^-2
C_SI = 299792458.0                # m s^-1
M_SUN_SI = 1.98847e30             # kg

# One-time mapping for a reference mass to a plot radius at t=0
BH_M_REF_SOLAR = 10.0             # reference BH mass in solar masses
BH_TARGET_RADIUS_FOR_REF = 0.28   # plot units for the reference mass

# Smooth, monotonic, small growth over the run
BH_GROWTH_MAX = 0.15              # max fractional increase (e.g., up to +15%)
BH_GROWTH_EMA_ALPHA = 0.25        # exponential moving average alpha (0<alpha<=1)

# Growth driver blend (nonnegative contributors only, to avoid shrink)
BH_GROW_W_HAWK = 0.6              # Hawking MI contribution
BH_GROW_W_ENTR_POS = 0.4          # positive dEntropy/dt contribution
BH_GROW_W_CURV = 0.2              # optional curvature mean contribution

# =============================================================================
# HELPERS
# =============================================================================

def frame_to_step_idx(frame, total_frames, time_steps):
    """Map animation frame index to nearest discrete quantum step index."""
    if total_frames <= 1:
        return 0
    return int(round((time_steps - 1) * frame / float(total_frames - 1)))

def safe_read_csv(path, index_col=None):
    """Read CSV if present, else return None."""
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, index_col=index_col)
    except Exception:
        return None

def norm01(x):
    """Normalize array to [0,1]. If invalid or flat, return zeros."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.zeros(0, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or hi - lo == 0.0:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)

def get_two_col_series(df, value_col_idx=1, T=TIME_STEPS):
    """Return a length-T series from a two-column dataframe [Step, Value]."""
    if df is None or df.shape[1] < value_col_idx + 1:
        return np.zeros(T, dtype=float)
    vals = df.iloc[:, value_col_idx].to_numpy(dtype=float)
    out = np.zeros(T, dtype=float)
    n = min(T, vals.shape[0])
    out[:n] = vals[:n]
    return out

def get_curv_grid(curv_lat_df, step_idx):
    """
    Return 3x3 curvature grid for the given step.
    Assumes rows are 9 cells in row-major order for each step column.
    """
    if curv_lat_df is None:
        return np.zeros((3, 3), dtype=float)
    try:
        vals = curv_lat_df.iloc[:9, step_idx].values.astype(float)
        return vals.reshape(3, 3)
    except Exception:
        return np.zeros((3, 3), dtype=float)

def central_gradient_unitvec(grid3):
    """Gradient at center cell (1,1) to bias swirl direction."""
    gx = 0.5 * (grid3[1, 2] - grid3[1, 0])
    gy = 0.5 * (grid3[2, 1] - grid3[0, 1])
    v = np.array([gx, gy], dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.array([0.0, 0.0])

def pos_to_cell(x, y, xmin=XLIM[0], xmax=XLIM[1], ymin=YLIM[0], ymax=YLIM[1]):
    """Map (x,y) to 3x3 cell indices (ix, iy) in {0,1,2}."""
    def to_idx(val, vmin, vmax):
        u = (val - vmin) / (vmax - vmin)
        return int(np.clip(np.floor(u * 3.0), 0, 2))
    ix = to_idx(x, xmin, xmax)
    iy = to_idx(y, ymin, ymax)
    return ix, iy

# =============================================================================
# LOAD QUANTUM OUTPUTS
# =============================================================================

hawk_df = safe_read_csv(CSV_FILES["hawking"])
entr_df = safe_read_csv(CSV_FILES["entropy"])
curv_lat_df = safe_read_csv(CSV_FILES["curv_lat"], index_col=0)

T = TIME_STEPS

hawk_t = get_two_col_series(hawk_df, value_col_idx=1, T=T)
entr_t = get_two_col_series(entr_df, value_col_idx=1, T=T)

# dEntropy/dt (discrete)
entr_rate_t = np.zeros_like(entr_t)
if T > 1:
    entr_rate_t[1:] = np.diff(entr_t)

# normalize base signals
hawk_n = norm01(hawk_t)
entr_rate_n = norm01(entr_rate_t)
entr_rate_pos = np.maximum(0.0, entr_rate_n)  # positive part only

# curvature mean per step (optional)
if curv_lat_df is not None:
    curv_mean = []
    for s in range(T):
        g = get_curv_grid(curv_lat_df, s)
        g_n = norm01(g)
        curv_mean.append(float(np.mean(g_n)))
    curv_mean = np.array(curv_mean, dtype=float)
    curv_mean_n = norm01(curv_mean)
else:
    curv_mean_n = np.zeros(T, dtype=float)

# =============================================================================
# GLOBAL PULL FOR STAR MOTION (not monotonic by design)
# =============================================================================

def pull_strength(step_idx):
    ps = PULL_W_ENTR_POS * entr_rate_pos[step_idx] + PULL_W_HAWK * hawk_n[step_idx]
    return max(0.0, float(ps))

# =============================================================================
# SCHWARZSCHILD MAPPING + MONOTONIC GROWTH
# =============================================================================

# One-time meter->plot radius mapping for the reference mass
BH_M_REF = BH_M_REF_SOLAR * M_SUN_SI
BH_RS_REF_M = 2.0 * G_SI * BH_M_REF / (C_SI ** 2)  # meters
BH_SCALE_M_TO_PLOT = BH_TARGET_RADIUS_FOR_REF / BH_RS_REF_M
BASE_EVENT_HORIZON_RADIUS = BH_TARGET_RADIUS_FOR_REF  # plot units at t=0

# Build a nonnegative growth driver in [0,1]
driver = (
    BH_GROW_W_HAWK * hawk_n
  + BH_GROW_W_ENTR_POS * entr_rate_pos
  + BH_GROW_W_CURV * curv_mean_n
)
driver = norm01(driver)

# Smooth step-to-step jitter with EMA, then cumulate to make it monotonic
ema = np.zeros(T, dtype=float)
if T > 0:
    ema[0] = driver[0]
for s in range(1, T):
    ema[s] = (1.0 - BH_GROWTH_EMA_ALPHA) * ema[s - 1] + BH_GROWTH_EMA_ALPHA * driver[s]

cum = np.cumsum(ema)
if cum.size > 0 and cum[-1] > 0.0:
    growth_phase = cum / cum[-1]   # maps to [0,1] over the run
else:
    growth_phase = np.zeros(T, dtype=float)

# Final monotonic growth factor: in [1, 1 + BH_GROWTH_MAX]
BH_GROWTH_FACTOR = 1.0 + BH_GROWTH_MAX * growth_phase

def bh_radius_at_step(step_idx):
    """Return event horizon radius at given step in plot units (monotonic)."""
    return BASE_EVENT_HORIZON_RADIUS * BH_GROWTH_FACTOR[step_idx]

# =============================================================================
# STAR FIELD INITIALIZATION
# =============================================================================

np.random.seed(42)

# scatter stars uniformly by area in annulus [R_MIN, R_MAX]
u = np.random.rand(N_STARS)
radii0 = np.sqrt((R_MAX**2 - R_MIN**2) * u + R_MIN**2)
angles0 = np.random.uniform(0, 2 * np.pi, size=N_STARS)
initial_positions = np.column_stack([radii0 * np.cos(angles0), radii0 * np.sin(angles0)])

ecc = np.clip(np.random.normal(loc=ECC_MEAN, scale=ECC_STD, size=N_STARS), 0.02, 0.55)
omega0 = np.random.uniform(ANG_VEL_RANGE[0], ANG_VEL_RANGE[1], size=N_STARS)
coupling = np.random.lognormal(mean=COUPLING_LOGN_MEAN, sigma=COUPLING_LOGN_SIGMA, size=N_STARS)
radial_drag = np.random.uniform(DRAG_RANGE[0], DRAG_RANGE[1], size=N_STARS)

orbital_params = [{"a": float(radii0[i]), "e": float(ecc[i]), "theta": float(angles0[i])} for i in range(N_STARS)]

np.random.seed(7)
star_sizes_base = np.random.lognormal(mean=4, sigma=0.5, size=N_STARS)
star_sizes_base = (star_sizes_base - star_sizes_base.min()) / (star_sizes_base.max() - star_sizes_base.min())
star_sizes_base = STAR_SIZE_MIN + star_sizes_base * (STAR_SIZE_MAX - STAR_SIZE_MIN)
star_rgba_base = np.column_stack([np.ones(N_STARS), np.ones(N_STARS), np.ones(N_STARS), np.ones(N_STARS)])

black_hole_pos = np.array([0.0, 0.0])
current_positions = initial_positions.copy()
inside_horizon = np.zeros(N_STARS, dtype=bool)

# =============================================================================
# BACKGROUND AND FIGURE
# =============================================================================

milkyway_array = None
if os.path.exists(BACKGROUND_IMG_PATH):
    try:
        milkyway_img = Image.open(BACKGROUND_IMG_PATH).resize((800, 800))
        milkyway_array = np.array(milkyway_img) / 255.0
    except Exception:
        milkyway_array = None

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(*XLIM)
ax.set_ylim(*YLIM)
ax.set_aspect("equal", "box")
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Black Hole with Entanglement-Driven Accretion", fontsize=14, color="white")
ax.set_facecolor("none")
fig.patch.set_facecolor("black")
if milkyway_array is not None:
    ax.imshow(milkyway_array, extent=(XLIM[0], XLIM[1], YLIM[0], YLIM[1]), zorder=0)

# =============================================================================
# ANIMATION UPDATE
# =============================================================================

def update(frame):
    step_idx = frame_to_step_idx(frame, TOTAL_FRAMES, TIME_STEPS)

    # reset axes and background
    ax.clear()
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal", "box")
    ax.set_xticks([])
    ax.set_yticks([])
    t = step_idx * DT
    ax.set_title(f"t = {t:.1f}", fontsize=14, color="white")
    ax.set_facecolor("none")
    fig.patch.set_facecolor("black")
    if milkyway_array is not None:
        ax.imshow(milkyway_array, extent=(XLIM[0], XLIM[1], YLIM[0], YLIM[1]), zorder=0)

    # curvature grid and normalized version
    grid3 = get_curv_grid(curv_lat_df, step_idx)
    grid3n = norm01(grid3)

    # swirl direction from central gradient
    grad_uv = central_gradient_unitvec(grid3n)

    # global pull and swirl for stellar motion
    PS = pull_strength(step_idx)
    swirl_gain = SWIRL_GAIN_SCALE * PS

    # black hole radius: monotonic, smoothed, small growth
    event_horizon_radius = bh_radius_at_step(step_idx)

    # visuals
    star_rgba = star_rgba_base.copy()
    star_sizes = star_sizes_base.copy()

    # evolve stars
    for i in range(N_STARS):
        if inside_horizon[i]:
            star_rgba[i, 3] = 0.0
            star_sizes[i] = 0.0
            continue

        p = orbital_params[i]
        a, e, th = p["a"], p["e"], p["theta"]

        # base inward decay and spin-up from global pull
        shrink = radial_drag[i] * coupling[i] * (0.6 + 0.8 * PS)
        omega = omega0[i] * (1.0 + 0.9 * PS)

        # local curvature modulation (optional)
        if USE_SPATIAL_CURVATURE_MOD and curv_lat_df is not None:
            ix, iy = pos_to_cell(current_positions[i][0], current_positions[i][1],
                                 xmin=XLIM[0], xmax=XLIM[1], ymin=YLIM[0], ymax=YLIM[1])
            local_curv = grid3n[iy, ix]
            local_mult = 1.0 + BETA_LOCAL * float(local_curv)
        else:
            local_mult = 1.0

        shrink *= local_mult
        omega *= (1.0 + 0.3 * (local_mult - 1.0))

        # swirl bias along gradient direction
        if np.any(grad_uv):
            pos_dir = current_positions[i] / (np.linalg.norm(current_positions[i]) + 1e-12)
            swirl = swirl_gain * float(np.clip(np.dot(pos_dir, grad_uv), -1.0, 1.0))
        else:
            swirl = 0.0

        th = th + omega + swirl

        # bounded eccentricity with mild variability
        e = float(np.clip(e * (1.0 - 0.08 * PS) + 0.01 * np.sin(0.7 * th), 0.02, 0.55))

        # update orbital state
        r = a * (1 - e**2) / (1 + e * np.cos(th))
        x, y = r * np.cos(th), r * np.sin(th)
        a = max(1e-4, a * (1.0 - shrink))
        current_positions[i] = [x, y]

        # absorption by horizon
        dist = np.hypot(x - black_hole_pos[0], y - black_hole_pos[1])
        if dist < event_horizon_radius:
            inside_horizon[i] = True
            star_rgba[i, 3] = 0.0
            star_sizes[i] = 0.0

        p["a"], p["e"], p["theta"] = a, e, th

    # draw stars
    ax.scatter(current_positions[:, 0], current_positions[:, 1],
               s=star_sizes, c=star_rgba, marker="o", zorder=2)

    # draw black hole disk in DATA units
    bh_disc = plt.Circle(black_hole_pos, event_horizon_radius,
                         color="black", alpha=1.0, zorder=3)
    ax.add_patch(bh_disc)

    # optional thin white ring
    ring = plt.Circle(black_hole_pos, event_horizon_radius,
                      color="white", alpha=0.26, fill=False, linewidth=2, zorder=4)
    ax.add_patch(ring)

    # legend
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", label="Stars",
               markerfacecolor="white", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Black Hole",
               markerfacecolor="black", markersize=5)
    ]
    ax.legend(handles=legend_elems, facecolor="black", edgecolor="white",
              loc="lower right", framealpha=0.7, fontsize=10)

    return []

# =============================================================================
# RUN
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Rendering animation with smooth, monotonic BH growth...")
    start = time.time()
    anim = animation.FuncAnimation(fig, update, frames=TOTAL_FRAMES,
                                   interval=FRAME_INTERVAL_MS, blit=False)
    anim.save(GIF_PATH, writer="pillow", fps=FPS)
    print(f"Saved GIF to {GIF_PATH} in {time.time() - start:.2f} s.")
    plt.close(fig)

if __name__ == "__main__":
    main()
