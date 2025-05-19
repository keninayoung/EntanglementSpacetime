import argparse
import os
import pandas as pd
from .graph_builder import build_graph, build_heisenberg_ham
from .curvature import compute_curvature
from .einstein_tensor import compute_einstein_tensor
from .entropy import compute_entropy
from .hawking_radiation import compute_hawking_radiation
from .visualization import save_entanglement_graph

def main():
    parser = argparse.ArgumentParser(description="Entanglement Spacetime Simulation")
    parser.add_argument("--Lx", type=int, default=3, help="Grid width")
    parser.add_argument("--Ly", type=int, default=3, help="Grid height")
    parser.add_argument("--steps", type=int, default=5, help="Number of time steps")
    parser.add_argument("--bond_dim", type=int, default=2, help="Bond dimension")
    parser.add_argument("--dt", type=float, default=0.1, help="Time step size")
    parser.add_argument("--hamiltonian", type=str, default="heisenberg", choices=["heisenberg", "ising"], help="Hamiltonian type")
    parser.add_argument("--output_dir", type=str, default="spacetime_outputs", help="Output directory")
    parser.add_argument("--approximate", action="store_true", help="Use approximate contraction")
    args = parser.parse_args()

    Lx, Ly = args.Lx, args.Ly
    n_sites = Lx * Ly
    bond_dim = args.bond_dim
    time_steps = args.steps
    dt = args.dt
    hamiltonian = args.hamiltonian
    output_dir = args.output_dir
    approximate = args.approximate

    os.makedirs(output_dir, exist_ok=True)

    print("Building initial PEPS...")
    peps = qtn.PEPS.rand(Lx=Lx, Ly=Ly, bond_dim=bond_dim, phys_dim=2, seed=42)

    if hamiltonian == "heisenberg":
        ham_terms = build_heisenberg_ham(Lx, Ly, J=1.0, cyclic=False)
    else:
        raise NotImplementedError("Ising Hamiltonian not yet implemented.")

    print("Starting time evolution...")
    mi_evolution = []
    graphs = []
    for t in range(time_steps):
        print(f"Time step {t+1}/{time_steps} (t={t*dt:.2f})...")
        if t > 0:
            for H_term, (site1, site2) in ham_terms:
                U = qu.expm(-1j * H_term * dt)
                peps.gate(U, where=(site1, site2), inplace=True)
                peps.compress_all(max_bond=bond_dim)
        G, df_mi = build_graph(peps, Lx, Ly, approximate)
        for i in G.nodes:
            x, y, _ = G.nodes[i]["pos"]
            G.nodes[i]["pos"] = (x, y, t)
        graphs.append(G)
        mi_evolution.append(df_mi)

    print("Computing outputs...")
    curvature_evolution = []
    hawking_mi = []
    einstein_approx = []
    entropies = []
    for t, (G, df_mi) in enumerate(zip(graphs, mi_evolution)):
        curv_t = compute_curvature(G)
        curvature_evolution.append(curv_t)
        entropies.append(compute_entropy(df_mi, n_sites))
        hawking_mi.append(compute_hawking_radiation(df_mi, Lx, Ly, n_sites))
        einstein_approx.append(compute_einstein_tensor(G, curv_t))
        save_entanglement_graph(G, t, dt, output_dir)

    print("Saving outputs...")
    curv_df = pd.DataFrame(
        {f"Step {t}": {f"{i}-{j}": v for (i, j), v in curv_t.items()}
        for t, curv_t in enumerate(curvature_evolution)}
    )
    curv_df.to_csv(os.path.join(output_dir, "curvature_evolution.csv"))
    hawking_df = pd.DataFrame({"Step": range(time_steps), "MI Across Horizon": hawking_mi})
    hawking_df.to_csv(os.path.join(output_dir, "hawking_radiation.csv"))
    einstein_df = pd.DataFrame(
        {f"Step {t}": einstein_t for t, einstein_t in enumerate(einstein_approx)}
    )
    einstein_df.to_csv(os.path.join(output_dir, "einstein_tensor.csv"))
    entropy_df = pd.DataFrame({"Step": range(time_steps), "Entropy": entropies})
    entropy_df.to_csv(os.path.join(output_dir, "entropy.csv"))
    print(f"Outputs saved in {output_dir}")

if __name__ == "__main__":
    main()