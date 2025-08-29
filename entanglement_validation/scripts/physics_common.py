# physics_common.py
# ASCII-only helpers and constants shared by the R8-R10 pipeline.

from __future__ import annotations
from pathlib import Path
import yaml
import math

# Physical constants (SI)
G_SI = 6.67430e-11          # m^3 kg^-1 s^-2
M_SUN_SI = 1.98847e30       # kg
C_SI = 2.99792458e8         # m s^-1
R_SUN_SI = 6.957e8          # m
AU_SI = 1.495978707e11      # m
ARCSEC_PER_RAD = 206264.80624709636

def load_cfg(conf_path: Path) -> dict:
    """
    Load YAML config. Same convention as R6.
    """
    with open(conf_path, "r") as f:
        return yaml.safe_load(f)

def default_conf_path(start_file: Path) -> Path:
    """
    Default to ../config.yaml relative to script file.
    """
    return (start_file.parent.parent / "config.yaml").resolve()

def ensure_out_dir(cfg: dict) -> Path:
    """
    Resolve and create paths.out_dir.
    """
    out_dir = Path(cfg["paths"]["out_dir"]).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def delta_phi_gr_per_orbit(G, M, c, a, e):
    """
    GR perihelion advance per orbit (radians), weak field.
    delta_phi = 6*pi*G*M / (a*(1-e^2)*c^2)
    """
    return (6.0 * math.pi * G * M) / (a * (1.0 - e * e) * c * c)

def arcsec_from_radians(angle_rad: float) -> float:
    return angle_rad * ARCSEC_PER_RAD

def radians_from_arcsec(angle_arcsec: float) -> float:
    return angle_arcsec / ARCSEC_PER_RAD
