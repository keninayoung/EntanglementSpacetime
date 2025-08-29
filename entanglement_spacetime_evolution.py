# =============================================================================
# EntanglementSpacetime Simulation Framework
# EDG: Quantum-Informed Predictor of Classical Spacetime Dynamics
# Authors: Kenneth Young, PhD; Paul Bransford
# Last updated: 2025-08-19
#
# Summary
# -------
# Simulates emergent spacetime geometry from quantum entanglement using a
# Projected Entangled Pair States (PEPS) ansatz evolved under a Heisenberg
# Hamiltonian. At each discrete time step, the code:
#   1) Applies two-site evolution gates to the PEPS.
#   2) Compresses to a target bond dimension.
#   3) Builds an entanglement graph from mutual information (MI).
#   4) Derives curvature-like, entropy, Hawking-like, and Einstein-like
#      diagnostics from the graph and MI.
#   5) Saves graphs and CSVs to drive downstream visualization and analysis.
# =============================================================================

import os
import time
import psutil
import numpy as np
import quimb as qu
import quimb.tensor as qtn
from graph_builder import build_graph, build_heisenberg_ham
from curvature import compute_curvature
from entropy import compute_entropy
from hawking_radiation import compute_hawking_radiation
from einstein_tensor import compute_einstein_tensor
from visualization import save_entanglement_graph
import pandas as pd

def custom_compress(peps, max_bond=2, max_iterations=3):
    """
    Optimized compression to enforce max_bond strictly.
    """
    for _ in range(max_iterations):
        prev_count = len(peps.tensors)
        peps.compress_all(max_bond=max_bond, cutoff=1e-16)  # stricter cutoff
        curr_count = len(peps.tensors)
        print(f"Compression iteration {_+1}/{max_iterations}, tensor count: {curr_count}, bond_dim={peps.max_bond()}")
        if curr_count <= peps.Lx * peps.Ly * 1.05 or curr_count == prev_count:
            break
    return peps

def run_simulation(
    Lx=3,
    Ly=3,
    bond_dim=2,
    time_steps=5,
    dt=0.1,
    J=1.0,
    approximate=False,
    use_gpu=True,
    output_dir="spacetime_outputs",
):
    """
    Run entanglement spacetime evolution simulation.

    Parameters:
    - Lx, Ly: Lattice dimensions.
    - bond_dim: Maximum bond dimension for PEPS.
    - time_steps: Number of time steps.
    - dt: Time step size.
    - J: Heisenberg coupling strength.
    - approximate: Use approximate MI computation.
    - use_gpu: Enable GPU computation with CuPy.
    - output_dir: Directory for output files.
    """
    n_sites = Lx * Ly
    memory = psutil.virtual_memory()
    print(f"Available memory: {memory.available / (1024 ** 3):.2f} GB")

    # Estimate memory for density matrix (informational only)
    est_density_mem = (2 ** n_sites) * 8 / (1024 ** 3)
    print(f"Estimated memory for density matrix: {est_density_mem:.2f} GB")

    # Check CuPy availability
    try:
        import cupy  # noqa: F401
        cupy_available = True
        print("CuPy is available. GPU acceleration can be used.")
    except ImportError:
        cupy_available = False
        use_gpu = False
        print("CuPy is not available. Using CPU (NumPy).")

    print(f"Computation mode: {'gpu' if use_gpu and cupy_available else 'cpu'}")

    print("Defining Heisenberg Hamiltonian...")
    H = build_heisenberg_ham(Lx, Ly, J=J, cyclic=False)

    print("Building initial PEPS...")
    peps = qtn.PEPS.rand(Lx, Ly, bond_dim=bond_dim, phys_dim=2, seed=42)
    norm = np.abs(peps.norm())
    if norm > 0:
        peps /= norm
    print(f"PEPS initialized: type={type(peps)}, Lx={Lx}, Ly={Ly}, bond_dim={bond_dim}")
    print(f"PEPS tensor shapes: {[t.shape for t in peps.tensors]}")
    print(f"Initial PEPS norm: {peps.norm():.6f}")

    print("Starting time evolution...")
    graphs = []
    mi_evolution = []

    expected_tensor_count = n_sites
    for t in range(time_steps):
        print(f"Time step {t+1}/{time_steps} (t={t*dt:.2f})...")
        if t > 0:
            for H_term, (site1, site2) in H:
                try:
                    U = qu.expm(-1j * H_term * dt)
                    print(f"Applying gate to sites {site1}, {site2}: U shape={U.shape}, tensor count before: {len(peps.tensors)}")
                    peps.gate(U, where=(site1, site2), inplace=True)
                    peps.compress_all(max_bond=bond_dim)
                    print(f"Post-compression bond_dim={peps.max_bond()}, tensor count: {len(peps.tensors)}")
                except Exception as e:
                    print(f"Error in gate application: {e}")
                    raise
            norm = np.abs(peps.norm())
            if norm > 0:
                peps /= norm
            tensor_count = len(peps.tensors)
            if tensor_count > expected_tensor_count * 1.05:
                print(f"Warning: Tensor count {tensor_count} exceeds expected {expected_tensor_count} by significant margin")
            print(f"PEPS norm after step {t+1}: {peps.norm():.6f}")
            print(f"PEPS tensor count after step {t+1}: {tensor_count}")
            print(f"PEPS tensor shapes after step {t+1}: {[t.shape for t in peps.tensors]}")

        print("Computing MI...")
        G, df_mi = build_graph(peps, Lx, Ly, approximate=approximate, use_gpu=use_gpu)
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

        for i in range(n_sites):
            if 'pos' not in G.nodes[i]:
                x = i % Lx
                y = i // Lx
                G.nodes[i]['pos'] = (x, y, t)

        save_entanglement_graph(G, t, dt, output_dir)
        print(
            f"Step {t}: Edges={G.number_of_edges()}, Nodes={G.number_of_nodes()}, "
            f"Max MI={df_mi['Mutual Information'].max() if not df_mi.empty else 0:.6f}, "
            f"Weight Range=[{df_mi['Mutual Information'].min() if not df_mi.empty else 0:.6f}, "
            f"{df_mi['Mutual Information'].max() if not df_mi.empty else 0:.6f}]"
        )

    print("Saving outputs...")
    os.makedirs(output_dir, exist_ok=True)

    # Save curvature with site pairs as rows
    curv_df = pd.DataFrame(
        {f"Step {t}": {f"{i}-{j}": v for (i, j), v in curv_t.items()}
         for t, curv_t in enumerate(curvature_evolution)}
    )
    curv_df.to_csv(os.path.join(output_dir, "curvature_evolution.csv"))

    pd.DataFrame({"Step": range(time_steps), "Entropy": entropies}).to_csv(
        os.path.join(output_dir, "entropy.csv"), index=False
    )
    pd.DataFrame({"Step": range(time_steps), "MI Across Horizon": hawking_mi}).to_csv(
        os.path.join(output_dir, "hawking_radiation.csv"), index=False
    )
    pd.DataFrame(einstein_approx).to_csv(os.path.join(output_dir, "einstein_tensor.csv"), index=False)

    # -------------------------------------------------------------------------
    # NEW: per-site curvature grid (n_sites rows x time_steps columns)
    # Derives a site scalar as mean absolute curvature over incident pairs.
    # This enables spatial anisotropy in downstream visuals.
    # -------------------------------------------------------------------------
    site_curv_over_time = np.zeros((n_sites, time_steps), dtype=float)
    for t, curv_t in enumerate(curvature_evolution):
        accum = [[] for _ in range(n_sites)]
        for (i, j), v in curv_t.items():
            v_abs = float(abs(v))
            accum[i].append(v_abs)
            accum[j].append(v_abs)
        for i in range(n_sites):
            site_curv_over_time[i, t] = np.mean(accum[i]) if accum[i] else 0.0

    curv_site_df = pd.DataFrame(
        site_curv_over_time,
        index=[f"site_{i}" for i in range(n_sites)],
        columns=[f"Step {t}" for t in range(time_steps)]
    )
    curv_site_df.to_csv(os.path.join(output_dir, "curvature_lattice.csv"))
    # -------------------------------------------------------------------------

    print(f"Outputs saved in {output_dir}")

if __name__ == "__main__":
    start_time = time.time()
    # 3x3 run
    run_simulation(
        Lx=3,
        Ly=3,
        bond_dim=2,
        time_steps=5,
        dt=0.1,
        J=1.0,
        approximate=False,
        use_gpu=True,
        output_dir="spacetime_outputs",
    )
    print(f"3x3 Simulation completed in {time.time() - start_time:.2f} seconds")
