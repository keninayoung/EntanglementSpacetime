#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# validation_r13p_joint_observational_fit.py
#
# R13+: Joint fit of EDG parameters (epsilon, L_q) to observational rows.
# ASCII-only, headless-safe, no external dependencies beyond numpy/pandas/mpl.
#
# Highlights
#   - Default weak-field mapping uses perihelion and gamma/Shapiro (optional).
#   - Strong-field mapping for EHT/GW scales with rs by default:
#       frac_pred = kappa * (L_q / rs)^p                 [dimensionless]
#       D_pred    = D_GR * (1 + kappa * (L_q / rs)^p)    [microas]
#   - Also supports a "fractional toy" mode using a user L_ref if desired.
#   - "auto" L_q grid: chooses a wide enough range using EHT input and rs.
#   - Outputs: posterior grid, best-fit, marginals, heatmap, constraints.md/tex
#     with Lq 95% UL, Delta AIC, Delta BIC, and a simple Delta log Z estimate.
#   - Writes an EHT sigma forecast CSV showing how tighter sigmas sharpen Lq95.
#
# Dataset CSV columns (for external ingestion)
#   dataset_id, dataset_type, observable, value, sigma, units, planet, meta_json
#   dataset_type in {perihelion, gamma, shapiro, eht_ring, eht_frac, gw_frac}
#
# Notes
#   - Periodic orbits: A0_pred = epsilon * GR_baseline (weak-field lever).
#   - gamma: epsilon anchors PPN gamma (gamma_pred = epsilon).
#   - shapiro: scales with gamma.
#   - eht_ring (absolute): D_pred = D_GR*(1 + kappa*(L_q/rs)^p).
#   - eht_frac (fractional): value is fractional offset from GR baseline,
#       frac_pred = kappa*(L_q/rs)^p. Put value ~ 0 and sigma ~ fractional err.
#   - gw_frac (fractional placeholder): same mapping as eht_frac.
#   - For EHT rows we compute rs = 2GM/c^2 from the given mass (Msun).
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Local helpers for config and output directory
from entanglement_validation.scripts.physics_common import (
    load_cfg, default_conf_path, ensure_out_dir
)

# -------------------------- Physical constants (SI) --------------------------

G_SI = 6.67430e-11        # m^3 kg^-1 s^-2
C_SI = 2.99792458e8       # m/s
M_SUN_SI = 1.98847e30     # kg

# -------------------------- Small utility helpers ---------------------------

def _fmt_float(x, sig=6):
    try:
        f = float(x)
    except Exception:
        return str(x)
    if f == 0.0:
        return "0"
    if abs(f) < 1e-3 or abs(f) >= 1e4:
        return "{:.{}e}".format(f, sig)
    return "{:.{}g}".format(f, sig)

def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print("[R13p] ERROR missing CSV:", path)
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        print("[R13p] ERROR reading {}: {}".format(path, e))
        return pd.DataFrame()

def _md_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_No data._\n"
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        vals = [str(row.get(c, "")) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"

def _latex_table(df: pd.DataFrame, caption="R13+ constraints", label="tab:r13p") -> str:
    if df is None or df.empty:
        return "% No data.\n"
    cols = list(df.columns)
    colspec = " | ".join(["l"] * len(cols))
    lines = []
    lines.append("\\begin{table}[h!]")
    lines.append("\\centering")
    lines.append("\\caption{%s}" % caption)
    lines.append("\\label{%s}" % label)
    lines.append("\\begin{tabular}{%s}" % colspec)
    lines.append("\\hline")
    lines.append(" & ".join(cols) + " \\\\")
    lines.append("\\hline")
    for _, row in df.iterrows():
        vals = [str(row.get(c, "")) for c in cols]
        lines.append(" & ".join(vals) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"

# -------------------------- Basic physics helpers ---------------------------

def schwarzschild_radius_m(M_kg: float) -> float:
    return 2.0 * G_SI * float(M_kg) / (C_SI * C_SI)

# -------------------------- EDG -> observable mappings ----------------------

def edg_to_gamma(epsilon: float, L_q: float, meta: Dict) -> float:
    return float(epsilon)

def perihelion_pred(epsilon: float, L_q: float, meta: Dict) -> float:
    gr = float(meta.get("gr_rad_per_orbit", np.nan))
    if not np.isfinite(gr):
        raise ValueError("perihelion meta requires gr_rad_per_orbit")
    return float(epsilon) * gr

def gamma_pred(epsilon: float, L_q: float, meta: Dict) -> float:
    return edg_to_gamma(epsilon, L_q, meta)

def shapiro_pred(epsilon: float, L_q: float, meta: Dict) -> float:
    base = float(meta.get("gr_pred_seconds", np.nan))
    if not np.isfinite(base):
        raise ValueError("shapiro meta requires gr_pred_seconds")
    gam = edg_to_gamma(epsilon, L_q, meta)
    return base * (gam / 1.0)

def eht_ring_pred_abs(epsilon: float, L_q: float, meta: Dict) -> float:
    """
    Absolute ring diameter prediction (microas):
      D_pred = D_GR * (1 + kappa * (L_q / rs)^p)
    Requires meta: D_GR_microas, rs_m, kappa, p
    """
    D_GR = float(meta.get("D_GR_microas", np.nan))
    rs = float(meta.get("rs_m", np.nan))
    kappa = float(meta.get("kappa", 0.0))
    power_p = float(meta.get("p", 2.0))
    if not np.isfinite(D_GR):
        raise ValueError("eht_ring meta requires D_GR_microas")
    if not np.isfinite(rs) or rs <= 0.0:
        raise ValueError("eht_ring meta requires positive rs_m")
    return D_GR * (1.0 + kappa * (float(L_q) / rs) ** power_p)

def frac_rs_pred(epsilon: float, L_q: float, meta: Dict) -> float:
    """
    Fractional strong-field prediction:
      frac_pred = kappa * (L_q / rs)^p
    For dataset types: eht_frac, gw_frac
    Requires meta: rs_m, kappa, p
    """
    rs = float(meta.get("rs_m", np.nan))
    kappa = float(meta.get("kappa", 0.0))
    power_p = float(meta.get("p", 2.0))
    if not np.isfinite(rs) or rs <= 0.0:
        raise ValueError("frac_rs_pred meta requires positive rs_m")
    return kappa * (float(L_q) / rs) ** power_p

# -------------------------- Likelihood assembly -----------------------------

def row_residual_and_sigma(epsilon: float, L_q: float, row: pd.Series) -> Tuple[float, float]:
    meta = {}
    if isinstance(row.get("meta_json", None), str) and row["meta_json"].strip():
        try:
            meta = json.loads(row["meta_json"])
        except Exception:
            meta = {}

    dtype = str(row.get("dataset_type", "")).strip().lower()
    val = float(row.get("value", np.nan))
    sig = float(row.get("sigma", np.nan))
    if not np.isfinite(val) or not np.isfinite(sig) or sig <= 0.0:
        raise ValueError("bad value/sigma in row: {}".format(row.get("dataset_id", "")))

    if dtype == "perihelion":
        pred = perihelion_pred(epsilon, L_q, meta)
    elif dtype == "gamma":
        pred = gamma_pred(epsilon, L_q, meta)
    elif dtype == "shapiro":
        pred = shapiro_pred(epsilon, L_q, meta)
    elif dtype == "eht_ring":
        pred = eht_ring_pred_abs(epsilon, L_q, meta)
    elif dtype in ("eht_frac", "gw_frac"):
        pred = frac_rs_pred(epsilon, L_q, meta)
    else:
        raise ValueError("unknown dataset_type: {}".format(dtype))

    resid = val - pred
    return float(resid), float(sig)

def joint_chi2(ext_df: pd.DataFrame, epsilon: float, L_q: float,
               eps_prior_mu: float = None, eps_prior_sigma: float = None) -> float:
    chi2 = 0.0
    # Data likelihood
    for _, row in ext_df.iterrows():
        try:
            r, s = row_residual_and_sigma(epsilon, L_q, row)
            chi2 += (r / s) ** 2
        except Exception as e:
            print("[R13p] WARN skipping row in chi2:", e)
            continue
    # Optional Gaussian prior on epsilon (e.g., Cassini gamma)
    if eps_prior_mu is not None and eps_prior_sigma is not None and eps_prior_sigma > 0.0:
        chi2 += ((epsilon - eps_prior_mu) / eps_prior_sigma) ** 2
    return float(chi2)

# -------------------------- Posterior grid and outputs ----------------------

def grid_posterior(ext_df: pd.DataFrame,
                   eps_grid: np.ndarray,
                   Lq_grid: np.ndarray,
                   eps_prior_mu: float = None,
                   eps_prior_sigma: float = None) -> pd.DataFrame:
    rows = []
    for e in eps_grid:
        for L in Lq_grid:
            c2 = joint_chi2(ext_df, e, L, eps_prior_mu, eps_prior_sigma)
            rows.append(dict(epsilon=float(e), L_q=float(L), chi2=float(c2)))
    df = pd.DataFrame(rows)
    df["delta_chi2"] = df["chi2"] - df["chi2"].min()
    df["loglike"] = -0.5 * df["delta_chi2"]
    # Uniform priors on grid by default; normalization below
    rel = np.exp(df["loglike"].to_numpy() - np.max(df["loglike"].to_numpy()))
    Z = float(np.sum(rel))
    df["posterior_norm"] = rel / Z if Z > 0.0 else rel
    return df

def marginals_from_grid(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    m_eps = df.groupby("epsilon")["posterior_norm"].sum().reset_index().rename(
        columns={"posterior_norm": "marginal"})
    m_Lq = df.groupby("L_q")["posterior_norm"].sum().reset_index().rename(
        columns={"posterior_norm": "marginal"})

    mu_eps = float(np.sum(m_eps["epsilon"] * m_eps["marginal"]))
    var_eps = float(np.sum(((m_eps["epsilon"] - mu_eps) ** 2) * m_eps["marginal"]))
    mu_Lq = float(np.sum(m_Lq["L_q"] * m_Lq["marginal"]))
    var_Lq = float(np.sum(((m_Lq["L_q"] - mu_Lq) ** 2) * m_Lq["marginal"]))

    summary = pd.DataFrame([
        dict(param="epsilon", mean=mu_eps, std=np.sqrt(max(0.0, var_eps))),
        dict(param="L_q", mean=mu_Lq, std=np.sqrt(max(0.0, var_Lq))),
    ])
    return summary, m_eps, m_Lq

def percentile_from_marginal(x: np.ndarray, w: np.ndarray, q: float) -> float:
    idx = np.argsort(x)
    x_sorted = x[idx]
    w_sorted = w[idx]
    cdf = np.cumsum(w_sorted) / np.sum(w_sorted)
    return float(np.interp(q, cdf, x_sorted))

def Lq_upper_95(m_Lq: pd.DataFrame) -> float:
    if m_Lq.empty:
        return float("nan")
    x = m_Lq["L_q"].to_numpy()
    w = m_Lq["marginal"].to_numpy()
    return percentile_from_marginal(x, w, 0.95)

def plot_heatmap(df: pd.DataFrame, out_png: Path):
    if df.empty:
        return
    pivot = df.pivot(index="L_q", columns="epsilon", values="posterior_norm")
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(pivot.values, aspect="auto", origin="lower",
                   extent=[pivot.columns.min(), pivot.columns.max(),
                           pivot.index.min(), pivot.index.max()])
    ax.set_xlabel("epsilon")
    ax.set_ylabel("L_q (meters)")
    ax.set_title("R13+: joint posterior (normalized)")
    fig.colorbar(im, ax=ax, label="posterior")
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("[R13p] wrote {}".format(out_png))

# -------------------------- Dynamic dataset builders ------------------------

def build_eht_row_abs(target: str,
                      D_GR_microas: float,
                      sigma_microas: float,
                      mass_solar: float,
                      kappa: float = 0.1,
                      p_power: float = 2.0) -> dict:
    M_kg = float(mass_solar) * M_SUN_SI
    rs_m = schwarzschild_radius_m(M_kg)
    meta = dict(D_GR_microas=float(D_GR_microas),
                rs_m=float(rs_m),
                kappa=float(kappa),
                p=float(p_power))
    row = dict(
        dataset_id="eht_{}_ring_abs".format(target.lower()),
        dataset_type="eht_ring",
        observable="ring_diam",
        value=float(D_GR_microas),
        sigma=float(sigma_microas),
        units="microas",
        planet="",
        meta_json=json.dumps(meta)
    )
    return row

def build_frac_row(target_id: str,
                   value_frac: float,
                   sigma_frac: float,
                   mass_solar: float,
                   kappa: float = 0.1,
                   p_power: float = 2.0,
                   dataset_type: str = "eht_frac") -> dict:
    M_kg = float(mass_solar) * M_SUN_SI
    rs_m = schwarzschild_radius_m(M_kg)
    meta = dict(rs_m=float(rs_m),
                kappa=float(kappa),
                p=float(p_power))
    row = dict(
        dataset_id="{}_frac".format(target_id),
        dataset_type=str(dataset_type),
        observable="fractional_shift",
        value=float(value_frac),
        sigma=float(sigma_frac),
        units="dimensionless",
        planet="",
        meta_json=json.dumps(meta)
    )
    return row

# -------------------------- Evidence proxies and model comparison -----------

def best_row_on_slice(df: pd.DataFrame, Lq_value: float) -> pd.Series:
    sl = df[np.isclose(df["L_q"], Lq_value)]
    if sl.empty:
        # pick closest L_q grid point
        i = int(np.argmin(np.abs(df["L_q"].to_numpy() - Lq_value)))
        return df.iloc[i]
    return sl.iloc[int(sl["loglike"].idxmax())]

def model_comparison_simple(df: pd.DataFrame, n_data: int, k_edg: int, k_gr: int):
    # EDG unrestricted min chi2
    chi2_edg = float(df["chi2"].min())
    # GR restricted at L_q = 0 (take best epsilon on that slice)
    gr_slice = df[df["L_q"] == df["L_q"].min()]  # assumes grid starts at 0
    if gr_slice.empty:
        # fallback to closest to 0
        Lq0 = float(df["L_q"].min())
        gr_slice = df[np.isclose(df["L_q"], Lq0)]
    chi2_gr = float(gr_slice["chi2"].min())

    # AIC/BIC
    AIC_edg = chi2_edg + 2 * k_edg
    BIC_edg = chi2_edg + k_edg * np.log(max(n_data, 1))
    AIC_gr = chi2_gr + 2 * k_gr
    BIC_gr = chi2_gr + k_gr * np.log(max(n_data, 1))

    dAIC = AIC_edg - AIC_gr
    dBIC = BIC_edg - BIC_gr
    # crude Delta logZ from BIC (Schwarz approx): logZ ~ -BIC/2
    dlogZ = -0.5 * (BIC_edg - BIC_gr)
    return dict(
        chi2_edg=chi2_edg, chi2_gr=chi2_gr,
        AIC_edg=AIC_edg, AIC_gr=AIC_gr, DeltaAIC=dAIC,
        BIC_edg=BIC_edg, BIC_gr=BIC_gr, DeltaBIC=dBIC,
        DeltaLogZ=dlogZ
    )

# -------------------------- L_q grid "auto" helper --------------------------

def auto_Lq_grid_from_eht(mu_frac: float, sig_frac: float,
                          rs_m: float, kappa: float, p_power: float,
                          n_sigma: float = 4.0,
                          num: int = 301) -> np.ndarray:
    """
    Choose an upper bound for L_q so that kappa*(L_q/rs)^p spans out to
    about |mu| + n_sigma*sig in fractional space.
    """
    span = abs(mu_frac) + n_sigma * abs(sig_frac)
    span = max(span, 1e-6)
    Lq_max = float(rs_m) * (span / max(kappa, 1e-12)) ** (1.0 / max(p_power, 1.0))
    Lq_max = max(Lq_max, 1.0)  # avoid zero
    return np.linspace(0.0, Lq_max, int(num))

# -------------------------- CLI / orchestrator ------------------------------

def main(argv=None):
    if argv is None:
        argv = []

    ap = argparse.ArgumentParser(
        description="R13+: Joint EDG fit (rs-scaled strong-field, with options)",
        add_help=True
    )
    ap.add_argument("--conf", default=str(default_conf_path(Path(__file__).resolve())),
                    help="Path to YAML config. Defaults to ../config.yaml.")
    ap.add_argument("--ext_csv", default=None,
                    help="External datasets CSV. If omitted, you can still add EHT/GW rows here.")

    # Grids; allow "auto" for L_q
    ap.add_argument("--eps_grid", default="0.98,1.02,401",
                    help="epsilon grid spec: start,stop,num (inclusive).")
    ap.add_argument("--Lq_grid", default="auto",
                    help="L_q grid spec: 'auto' or start,stop,num (inclusive).")

    # Priors and weak-field toggles
    ap.add_argument("--use_gamma", action="store_true",
                    help="Apply Gaussian prior on epsilon from Cassini-like gamma.")
    ap.add_argument("--gamma_mu", type=float, default=1.0,
                    help="Mean of epsilon prior if --use_gamma is set.")
    ap.add_argument("--gamma_sigma", type=float, default=2.3e-5,
                    help="Sigma of epsilon prior if --use_gamma is set.")
    ap.add_argument("--use_shapiro", action="store_true",
                    help="If you are ingesting a Shapiro row, keep True for mapping consistency.")

    # Strong-field rows (rs-scaled fractional or absolute)
    ap.add_argument("--add_eht", default="none", choices=["none", "sgrA", "m87"],
                    help="Append an EHT row.")
    ap.add_argument("--eht_mode", default="fractional", choices=["fractional", "absolute"],
                    help="EHT mapping: fractional uses frac_pred, absolute uses diameter.")
    ap.add_argument("--eht_mu_frac", type=float, default=0.0,
                    help="EHT fractional offset mean (dimensionless).")
    ap.add_argument("--eht_sigma_frac", type=float, default=0.02,
                    help="EHT fractional offset sigma (dimensionless).")
    ap.add_argument("--eht_DGR", type=float, default=None,
                    help="EHT GR diameter [microas] if absolute mode.")
    ap.add_argument("--eht_sigma_abs", type=float, default=None,
                    help="EHT sigma [microas] if absolute mode.")
    ap.add_argument("--eht_mass", type=float, default=None,
                    help="BH mass in solar masses. Defaults per target if None.")
    ap.add_argument("--eht_kappa", type=float, default=0.1)
    ap.add_argument("--eht_p", type=float, default=2.0)

    # GW fractional toy
    ap.add_argument("--add_gw", action="store_true",
                    help="Append a GW fractional row (rs-scaled).")
    ap.add_argument("--gw_mu_frac", type=float, default=0.0)
    ap.add_argument("--gw_sigma_frac", type=float, default=0.02)
    ap.add_argument("--gw_mass", type=float, default=30.0,
                    help="Reference BH mass in solar masses for rs in GW toy.")
    ap.add_argument("--gw_kappa", type=float, default=0.1)
    ap.add_argument("--gw_p", type=float, default=2.0)

    ap.add_argument("--check", action="store_true",
                    help="Print shapes and best-fit.")

    args, extras = ap.parse_known_args(argv)
    if extras:
        print("[R13p] INFO ignoring unknown args:", extras)

    cfg = load_cfg(Path(args.conf))
    out_dir = ensure_out_dir(cfg)

    # 1) Load external CSV if provided
    if args.ext_csv:
        ext_df = _read_csv(Path(args.ext_csv))
    else:
        ext_df = pd.DataFrame()

    # 2) Append EHT row if requested
    rs_for_auto = None
    if args.add_eht and args.add_eht.lower() != "none":
        target = args.add_eht
        if target == "sgrA":
            Msol_default = 4.1e6
            DGR_default, sig_abs_default = 51.8, 3.0
        else:
            Msol_default = 6.5e9
            DGR_default, sig_abs_default = 42.0, 3.0

        mass_solar = float(args.eht_mass) if args.eht_mass is not None else Msol_default
        M_kg = mass_solar * M_SUN_SI
        rs_for_auto = schwarzschild_radius_m(M_kg)

        if args.eht_mode == "fractional":
            row = build_frac_row(
                target_id="eht_{}".format(target.lower()),
                value_frac=float(args.eht_mu_frac),
                sigma_frac=float(args.eht_sigma_frac),
                mass_solar=mass_solar,
                kappa=float(args.eht_kappa),
                p_power=float(args.eht_p),
                dataset_type="eht_frac"
            )
            ext_df = pd.concat([ext_df, pd.DataFrame([row])], ignore_index=True)
            print("[R13p] INFO appended EHT FRACTIONAL row for {} (value={}, sigma={}).".format(
                target, _fmt_float(args.eht_mu_frac), _fmt_float(args.eht_sigma_frac)))
        else:
            D_GR = float(args.eht_DGR) if args.eht_DGR is not None else DGR_default
            sig_abs = float(args.eht_sigma_abs) if args.eht_sigma_abs is not None else sig_abs_default
            row = build_eht_row_abs(
                target=target,
                D_GR_microas=D_GR,
                sigma_microas=sig_abs,
                mass_solar=mass_solar,
                kappa=float(args.eht_kappa),
                p_power=float(args.eht_p)
            )
            ext_df = pd.concat([ext_df, pd.DataFrame([row])], ignore_index=True)
            print("[R13p] INFO appended EHT ABSOLUTE row for {} (D_GR={}, sigma={}).".format(
                target, _fmt_float(D_GR), _fmt_float(sig_abs)))

    # 3) Append GW fractional toy if requested
    if args.add_gw:
        row = build_frac_row(
            target_id="gw_qnm",
            value_frac=float(args.gw_mu_frac),
            sigma_frac=float(args.gw_sigma_frac),
            mass_solar=float(args.gw_mass),
            kappa=float(args.gw_kappa),
            p_power=float(args.gw_p),
            dataset_type="gw_frac"
        )
        ext_df = pd.concat([ext_df, pd.DataFrame([row])], ignore_index=True)
        print("[R13p] INFO appended GW QNM FRACTIONAL row (value={}, sigma={}).".format(
            _fmt_float(args.gw_mu_frac), _fmt_float(args.gw_sigma_frac)))

    if ext_df.empty:
        print("[R13p] nothing to do; dataset is empty.")
        return

    # 4) Parse grids (epsilon is explicit, L_q can be "auto")
    def _parse_grid(spec: str) -> np.ndarray:
        parts = [p.strip() for p in str(spec).split(",")]
        if len(parts) != 3:
            raise ValueError("grid spec must be start,stop,num")
        start, stop, num = float(parts[0]), float(parts[1]), int(parts[2])
        return np.linspace(start, stop, num)

    eps_grid = _parse_grid(args.eps_grid)

    if isinstance(args.Lq_grid, str) and args.Lq_grid.strip().lower() == "auto":
        # Find an EHT fractional row if present to set "auto" range
        eht_rows = ext_df[ext_df["dataset_type"] == "eht_frac"]
        if rs_for_auto is not None and not eht_rows.empty:
            mu_frac = float(eht_rows["value"].iloc[0])
            sig_frac = float(eht_rows["sigma"].iloc[0])
            kappa = json.loads(eht_rows["meta_json"].iloc[0]).get("kappa", 0.1)
            p_power = json.loads(eht_rows["meta_json"].iloc[0]).get("p", 2.0)
            Lq_grid = auto_Lq_grid_from_eht(mu_frac, sig_frac, rs_for_auto, kappa, p_power,
                                            n_sigma=4.0, num=401)
            print("[R13p] INFO auto L_q grid max ~ {}".format(_fmt_float(Lq_grid.max())))
        else:
            # fallback
            Lq_grid = np.linspace(0.0, 1.0e8, 401)
    else:
        Lq_grid = _parse_grid(args.Lq_grid)

    # 5) Build posterior on the grid with optional epsilon prior
    eps_mu = float(args.gamma_mu) if args.use_gamma else None
    eps_sig = float(args.gamma_sigma) if args.use_gamma else None
    post = grid_posterior(ext_df, eps_grid, Lq_grid, eps_mu, eps_sig)

    grid_path = out_dir / "R13p_posterior_grid.csv"
    post.to_csv(grid_path, index=False)
    print("[R13p] wrote {}".format(grid_path))

    # 6) 1D marginals and summary
    summary_df, m_eps, m_Lq = marginals_from_grid(post)
    Lq_u95 = Lq_upper_95(m_Lq)

    marg_path = out_dir / "R13p_marginals.csv"
    summary_df.to_csv(marg_path, index=False)
    print("[R13p] wrote {}".format(marg_path))

    # 7) Best fit
    best_idx = int(post["posterior_norm"].idxmax())
    best = post.iloc[best_idx]
    bf_path = out_dir / "R13p_best_fit.txt"
    with open(bf_path, "w") as f:
        f.write("epsilon_best = {}\n".format(_fmt_float(best["epsilon"])))
        f.write("L_q_best    = {}\n".format(_fmt_float(best["L_q"])))
        f.write("chi2_best   = {}\n".format(_fmt_float(best["chi2"])))
        f.write("Lq_95_upper = {}\n".format(_fmt_float(Lq_u95)))
    print("[R13p] wrote {}".format(bf_path))

    # 8) Heatmap
    heatmap_png = out_dir / "R13p_posterior_heatmap.png"
    plot_heatmap(post, heatmap_png)

    # 9) Simple evidence proxies: EDG vs GR (L_q = 0 slice)
    # n_data = number of independent rows (plus optional epsilon prior treated as data-like)
    n_rows = int(ext_df.shape[0])
    n_data = n_rows + (1 if args.use_gamma else 0)
    # k parameters: EDG has 2 (epsilon, L_q); GR has 1 (epsilon)
    mc = model_comparison_simple(post, n_data=n_data, k_edg=2, k_gr=1)

    # 10) EHT sigma forecast table (if an EHT fractional row is present)
    forecast_df = pd.DataFrame()
    eht_rows = ext_df[ext_df["dataset_type"] == "eht_frac"]
    if rs_for_auto is not None and not eht_rows.empty:
        mu_frac = float(eht_rows["value"].iloc[0])
        sig_frac = float(eht_rows["sigma"].iloc[0])
        meta = json.loads(eht_rows["meta_json"].iloc[0])
        kappa = float(meta.get("kappa", 0.1))
        p_power = float(meta.get("p", 2.0))

        mults = [1.0, 0.75, 0.5, 0.33, 0.25, 0.2, 0.1]
        rows = []
        for m in mults:
            span = abs(mu_frac) + 2.0 * (m * sig_frac)
            Lq95_est = float(rs_for_auto) * (span / max(kappa, 1e-12)) ** (1.0 / max(p_power, 1.0))
            rows.append(dict(sigma_mult=m, Lq95_est=Lq95_est))
        forecast_df = pd.DataFrame(rows)
        forecast_csv = out_dir / "R13p_eht_sigma_forecast.csv"
        forecast_df.to_csv(forecast_csv, index=False)
        print("[R13p] wrote {}".format(forecast_csv))

    # 11) Markdown / LaTeX summaries
    md_lines = []
    md_lines.append("# R13+: Joint EDG Fit")
    md_lines.append("")
    md_lines.append("Best fit:")
    md_lines.append("- epsilon_best = {}".format(_fmt_float(best["epsilon"])))
    md_lines.append("- L_q_best = {} m".format(_fmt_float(best["L_q"])))
    md_lines.append("- chi2_best = {}".format(_fmt_float(best["chi2"])))
    md_lines.append("")
    md_lines.append("1D posteriors (means and std on the discrete grid):")
    md_lines.append(_md_table(summary_df))
    md_lines.append("")
    md_lines.append("L_q 95pct upper limit: {} m".format(_fmt_float(Lq_u95)))
    md_lines.append("")
    md_lines.append("Model comparison (EDG vs GR with L_q=0):")
    mc_df = pd.DataFrame([mc])
    md_lines.append(_md_table(mc_df[["chi2_edg", "chi2_gr", "DeltaAIC", "DeltaBIC", "DeltaLogZ"]]))
    if not forecast_df.empty:
        md_lines.append("")
        md_lines.append("EHT sigma multiplier forecast (fractional mode):")
        md_lines.append(_md_table(forecast_df))

    with open(out_dir / "R13p_constraints.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print("[R13p] wrote", out_dir / "R13p_constraints.md")

    tex_lines = []
    tex_lines.append("\\section{R13+: Joint EDG Fit}")
    tex_lines.append("Best fit: $\\epsilon = %s$, $L_q = %s\\,\\mathrm{m}$, $\\chi^2 = %s$."
                     % (_fmt_float(best["epsilon"]), _fmt_float(best["L_q"]), _fmt_float(best["chi2"])))
    tex_lines.append(_latex_table(summary_df, caption="Discrete-grid 1D posteriors", label="tab:r13p_marg"))
    tex_lines.append(_latex_table(pd.DataFrame([mc])[["chi2_edg","chi2_gr","DeltaAIC","DeltaBIC","DeltaLogZ"]],
                                  caption="Model comparison proxies (EDG vs GR)", label="tab:r13p_model"))
    with open(out_dir / "R13p_constraints.tex", "w") as f:
        f.write("\n".join(tex_lines) + "\n")
    print("[R13p] wrote", out_dir / "R13p_constraints.tex")

    if args.check:
        print("[R13p] shapes: ext_df={}, post={}".format(getattr(ext_df, "shape", None), getattr(post, "shape", None)))
        print("[R13p] best:", dict(epsilon=float(best["epsilon"]), L_q=float(best["L_q"]), chi2=float(best["chi2"])))
        print("[R13p] Lq 95pct upper limit:", _fmt_float(Lq_u95))


def run(conf_path=None, ext_csv=None, eps_grid=None, Lq_grid=None,
        use_gamma=False, gamma_mu=1.0, gamma_sigma=2.3e-5,
        add_eht=None, eht_mode="fractional",
        eht_mu_frac=0.0, eht_sigma_frac=0.02, eht_DGR=None, eht_sigma_abs=None,
        eht_mass=None, eht_kappa=0.1, eht_p=2.0,
        add_gw=False, gw_mu_frac=0.0, gw_sigma_frac=0.02, gw_mass=30.0, gw_kappa=0.1, gw_p=2.0,
        check=False):
    argv = []
    if conf_path:
        argv += ["--conf", str(conf_path)]
    if ext_csv:
        argv += ["--ext_csv", str(ext_csv)]
    if eps_grid:
        argv += ["--eps_grid", str(eps_grid)]
    if Lq_grid:
        argv += ["--Lq_grid", str(Lq_grid)]

    if use_gamma:
        argv += ["--use_gamma", "--gamma_mu", str(float(gamma_mu)), "--gamma_sigma", str(float(gamma_sigma))]

    if add_eht and add_eht.lower() != "none":
        argv += ["--add_eht", str(add_eht), "--eht_mode", str(eht_mode)]
        if eht_mode == "fractional":
            argv += ["--eht_mu_frac", str(float(eht_mu_frac)),
                     "--eht_sigma_frac", str(float(eht_sigma_frac))]
        else:
            if eht_DGR is not None:       argv += ["--eht_DGR", str(float(eht_DGR))]
            if eht_sigma_abs is not None: argv += ["--eht_sigma_abs", str(float(eht_sigma_abs))]
        if eht_mass is not None:  argv += ["--eht_mass", str(float(eht_mass))]
        argv += ["--eht_kappa", str(float(eht_kappa)), "--eht_p", str(float(eht_p))]

    if add_gw:
        argv += ["--add_gw",
                 "--gw_mu_frac", str(float(gw_mu_frac)),
                 "--gw_sigma_frac", str(float(gw_sigma_frac)),
                 "--gw_mass", str(float(gw_mass)),
                 "--gw_kappa", str(float(gw_kappa)),
                 "--gw_p", str(float(gw_p))]

    if check:
        argv += ["--check"]
    return main(argv)


if __name__ == "__main__":
    main([])
