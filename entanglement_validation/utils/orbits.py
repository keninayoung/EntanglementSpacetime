# orbits.py
# Simple 2D velocity-Verlet integrator for central-force orbits.
import math
from dataclasses import dataclass
from typing import Dict
import numpy as np

@dataclass
class OrbitIC:
    r0: float
    v0: float
    k0: float
    e: float = 0.0

def velocity_verlet_orbit(ic: OrbitIC, dt: float, steps: int, k_time=None) -> Dict[str, np.ndarray]:
    x = np.array([ic.r0, 0.0], dtype=float); v = np.array([0.0, ic.v0], dtype=float); t = 0.0
    T = np.zeros(steps + 1); X = np.zeros(steps + 1); Y = np.zeros(steps + 1)
    VX = np.zeros(steps + 1); VY = np.zeros(steps + 1); R = np.zeros(steps + 1); TH = np.zeros(steps + 1)
    def accel(t_local, xvec):
        r = np.linalg.norm(xvec)
        if r == 0.0: return np.zeros_like(xvec)
        k = ic.k0 if k_time is None else k_time(t_local)
        return -k * xvec / (r ** 3)
    T[0]=t; X[0]=x[0]; Y[0]=x[1]; VX[0]=v[0]; VY[0]=v[1]; R[0]=np.linalg.norm(x); TH[0]=math.atan2(x[1], x[0])
    a = accel(t, x)
    for n in range(steps):
        x_new = x + v*dt + 0.5*a*(dt**2)
        t_new = t + dt
        a_new = accel(t_new, x_new)
        v_new = v + 0.5*(a + a_new)*dt
        x, v, a, t = x_new, v_new, a_new, t_new
        T[n+1]=t; X[n+1]=x[0]; Y[n+1]=x[1]; VX[n+1]=v[0]; VY[n+1]=v[1]; R[n+1]=np.linalg.norm(x); TH[n+1]=math.atan2(x[1], x[0])
    return {"t":T, "x":X, "y":Y, "vx":VX, "vy":VY, "r":R, "theta":TH}

def kepler_period(r0: float, k0: float) -> float:
    return 2 * math.pi * math.sqrt((r0 ** 3) / k0)

def circular_speed(r0: float, k0: float) -> float:
    return math.sqrt(k0 / r0)

def osculating_perihelion(theta: np.ndarray, r: np.ndarray) -> float:
    mins = [i for i in range(1, len(r)-1) if r[i] < r[i-1] and r[i] < r[i+1]]
    if len(mins) < 3: return 0.0
    ang = np.unwrap(np.array([theta[i] for i in mins]))
    return float(np.mean(np.diff(ang)))
