# Smoke test: Kepler period recovery.
import numpy as np
from utils.orbits import OrbitIC, velocity_verlet_orbit, circular_speed, kepler_period

def test_kepler_period_recovery():
    r0, k0, dt = 3.0, 100.0, 0.01
    T_kep = kepler_period(r0, k0)
    steps = int(5 * T_kep / dt)
    ic = OrbitIC(r0=r0, v0=circular_speed(r0, k0), k0=k0)
    out = velocity_verlet_orbit(ic, dt=dt, steps=steps)
    r, t = out["r"], out["t"]
    peaks = [i for i in range(1, len(r) - 1) if r[i] > r[i - 1] and r[i] > r[i + 1]]
    assert len(peaks) >= 2
    T_num = np.mean(np.diff(t[np.array(peaks)]))
    assert abs(T_num - T_kep) / T_kep < 1e-2
