# demo_synth.py
# Generate synthetic MI CSVs on a small grid for dry runs.
import os, csv, math, random

def synthetic_mi_on_grid(Lx=3, Ly=3, t_steps=5, seed=0, out_dir="synth_outputs"):
    random.seed(seed); os.makedirs(out_dir, exist_ok=True)
    def idx(x, y): return y * Lx + x
    for t in range(t_steps):
        fname = os.path.join(out_dir, f"mi_t{t}.csv")
        with open(fname, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f); wr.writerow(["t", "i", "j", "Iij"])
            for y in range(Ly):
                for x in range(Lx):
                    i = idx(x, y)
                    for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                        xn, yn = x+dx, y+dy
                        if 0 <= xn < Lx and 0 <= yn < Ly:
                            j = idx(xn, yn)
                            if i < j:
                                base = 0.6 + 0.2 * math.cos(0.7 * t)
                                noise = 0.02 * (random.random() - 0.5)
                                I = max(1e-6, base + noise) * (0.95 ** t)
                                wr.writerow([t, i, j, I])
    return out_dir

if __name__ == "__main__":
    out = synthetic_mi_on_grid(); print("Wrote synthetic MI CSVs to", out)
