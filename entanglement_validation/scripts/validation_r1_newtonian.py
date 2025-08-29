# R1: Newtonian limit sanity with quiet entanglement field.
import os, argparse, yaml
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from entanglement_validation.utils.orbits import (
    OrbitIC, velocity_verlet_orbit, circular_speed, kepler_period
)

def run(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    out_dir = cfg["paths"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    r0 = cfg["orbits"]["r0"]
    k0 = cfg["orbits"]["k0"]
    dt_list = cfg["orbits"]["dt_list"]
    periods = cfg["orbits"]["periods"]

    T_kep = kepler_period(r0, k0)
    err_T = []
    for dt in dt_list:
        steps = int(periods * T_kep / dt)
        ic = OrbitIC(r0=r0, v0=circular_speed(r0, k0), k0=k0)
        out = velocity_verlet_orbit(ic, dt=dt, steps=steps, k_time=None)
        r = out["r"]; t = out["t"]
        peaks = [i for i in range(1, len(r) - 1) if r[i] > r[i - 1] and r[i] > r[i + 1]]
        if len(peaks) >= 2:
            T_num = np.mean(np.diff(t[np.array(peaks)]))
            err_T.append(abs(T_num - T_kep) / T_kep)
        else:
            err_T.append(np.nan)

    plt.figure()
    plt.loglog(dt_list, err_T, marker="o")
    plt.xlabel("Time step dt")
    plt.ylabel("Relative period error")
    plt.title("R1: Newtonian limit (quiet entanglement field)")
    plt.grid(True, which="both", ls=":")
    fig_path = os.path.join(out_dir, "R1_newtonian_period_error.png")
    plt.savefig(fig_path, dpi=180, bbox_inches="tight")
    print("[R1] Wrote {}".format(fig_path))

def main():
    default_conf = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", default=default_conf, help="Path to config.yaml")
    args = ap.parse_args()
    run(args.conf)

if __name__ == "__main__":
    main()
