"""
EntanglementSpacetime Simulation Framework
=========================================

Author: Kenneth Young, PhD
Created: May 2025

Purpose:
--------
This program simulates the emergence of spacetime from quantum entanglement using
time-evolved tensor networks. It employs a Projected Entangled Pair States (PEPS)
tensor network evolved under a Heisenberg Hamiltonian to construct an entanglement
graph, where spacetime distances are defined as d(i, j) ~ -log I(i:j), with I(i:j)
being the mutual information between quantum sites. The framework computes discrete
curvature, approximates the Einstein tensor, and analyzes holographic entropy and
black hole dynamics, providing a computational approach to study quantum gravity and
holography.

This code supports both single-GPU (Windows/Linux) and multi-GPU (Linux only)
parallelization, with tensor contraction optimization using Optuna when KaHyPar is
unavailable.

Publication:
------------
For methodology and results, see:
"Entanglement-Driven Emergent Spacetime with Time-Evolved Tensor Networks"
by Kenneth Young, PhD (2025).

Usage:
------
Run a simulation with the following command:
    python3 -m emergent_spacetime.cli --Lx 3 --Ly 3 --steps 5 --hamiltonian heisenberg --use_gpu True

Requirements:
-------------
- Python 3.9+
- CUDA 12.6 (for GPU support)
- Minimum 16 GB RAM for 4x4 grids
- Linux for multi-GPU parallelization (Windows supports single-GPU or CPU)

See README.md for detailed installation and usage instructions.
"""

import pandas as pd
import numpy as np
import os
import platform
import quimb as qu
import quimb.tensor as qtn
from tqdm import tqdm
import psutil
from itertools import combinations
from scipy.sparse import identity, csr_matrix
from scipy.linalg import eigvalsh
import multiprocessing as mp
from functools import partial

# Check if kahypar is available
try:
    import kahypar
    kahypar_available = True
    print("KaHyPar is available. Using default HyperOptimizer for cotengra.")
except ImportError:
    kahypar_available = False
    print("KaHyPar is not available.")

# Check if optuna is available
try:
    import optuna
    optuna_available = True
    print("Optuna is available. Using Optuna-based HyperOptimizer for cotengra.")
except ImportError:
    optuna_available = False
    print("Optuna is not available. Falling back to greedy method for cotengra optimization.")

# Import cotengra to set the optimizer
import cotengra

# Set the cotengra optimizer based on availability
if kahypar_available:
    cotengra_optimizer = cotengra.HyperOptimizer()
elif optuna_available:
    cotengra_optimizer = cotengra.HyperOptimizer(methods=['optuna'])
else:
    cotengra_optimizer = 'greedy'

# Import dask for CPU parallelization
import dask

# Import cupy for GPU support
import cupy as cp

# Import dask-cuda only on Linux for multi-GPU support
is_windows = platform.system() == "Windows"
if not is_windows:
    try:
        from dask_cuda import LocalCUDACluster
        from dask.distributed import Client
        print("Dask-CUDA is available on Linux. Multi-GPU acceleration can be used if enabled.")
        dask_cuda_available = True
    except ImportError:
        print("Dask-CUDA is not available. Falling back to single-GPU or CPU.")
        dask_cuda_available = False
else:
    print("Dask-CUDA is not supported on Windows. Using single-GPU or CPU.")
    dask_cuda_available = False

try:
    print("CuPy is available. GPU acceleration can be used if enabled.")
    cupy_available = True
except ImportError:
    print("CuPy is not available. Falling back to NumPy (CPU).")
    cp = np
    cupy_available = False

from graph_builder import build_graph, build_heisenberg_ham
from curvature import compute_curvature
from einstein_tensor import compute_einstein_tensor
from entropy import compute_entropy
from hawking_radiation import compute_hawking_radiation
from visualization import save_entanglement_graph

def run_simulation(Lx=3, Ly=3, bond_dim=2, time_steps=5, dt=0.1, output_dir="spacetime_outputs", approximate=False, use_gpu=True):
    n_sites = Lx * Ly

    # Set up Dask CUDA cluster for multi-GPU if use_gpu=True and on Linux
    client = None
    if use_gpu and cupy_available and not is_windows and dask_cuda_available:
        try:
            cluster = LocalCUDACluster()
            client = Client(cluster)
            print("Dask CUDA Cluster initialized for multi-GPU computation.")
        except Exception as e:
            print(f"Failed to initialize Dask CUDA Cluster: {e}. Falling back to single-GPU or CPU.")
            client = None

    # If on Windows and use_gpu=True, use single-GPU without dask-cuda
    # If use_gpu=False or GPU not available, fall back to CPU
    if use_gpu and cupy_available:
        if is_windows or client is None:
            print("Using single-GPU computation with CuPy.")
        # Already set up for multi-GPU if client is not None
    else:
        use_gpu = False
        print("GPU not available or disabled. Falling back to CPU.")

    # --- Memory Check ---
    available_memory = psutil.virtual_memory().available / (1024 ** 3)  # GB
    required_memory = (2 ** n_sites) ** 2 * 16 / (1024 ** 3)  # Approx GB
    print(f"Available memory: {available_memory:.2f} GB")
    print(f"Estimated memory for density matrix: {required_memory:.2f} GB")
    if approximate or available_memory < required_memory * 1.5 or n_sites >= 16:
        approximate = True
        print("Using approximate contraction due to memory constraints.")

    # --- Define Heisenberg Hamiltonian ---
    print("Defining Heisenberg Hamiltonian...")
    ham_terms = build_heisenberg_ham(Lx, Ly, J=1.0, cyclic=False)

    # --- Initialize PEPS ---
    print("Building initial PEPS...")
    peps = qtn.PEPS.rand(Lx=Lx, Ly=Ly, bond_dim=bond_dim, phys_dim=2, seed=42)

    # --- Entropy and MI Functions ---
    def entropy(rho, xp=np):
        if xp is cp and cupy_available:
            try:
                rho_np = cp.asnumpy(rho)
            except Exception as e:
                print(f"CuPy error in entropy: {e}. Falling back to NumPy.")
                xp = np
                rho_np = rho
        else:
            rho_np = rho
        vals = eigvalsh(rho_np)
        vals = vals[vals > 1e-20]
        return -xp.sum(vals * xp.log(vals)) if len(vals) > 0 else 0.0

    def mutual_info(rho_ij, rho_i, rho_j, xp=np):
        mi = entropy(rho_i, xp) + entropy(rho_j, xp) - entropy(rho_ij, xp)
        return max(0.0, float(mi))

    def compute_single_site(rho_full_np, dims, i, xp=np):
        rho_single = qu.partial_trace(rho_full_np, dims, keep=[i])
        rho_single = xp.array(rho_single)
        trace_i = xp.trace(rho_single)
        if not xp.isclose(trace_i, 1.0, rtol=1e-5):
            print(f"Warning: rho_single[{i}] trace={float(trace_i)}")
        return rho_single

    def compute_mi_pair(peps, n_sites, i, j, rho_full_np, rho_single, approximate=False, xp=np):
        rho_ij = qu.partial_trace(rho_full_np, dims, keep=[i, j])
        rho_ij = xp.array(rho_ij)
        rho_i = rho_single[i]
        rho_j = rho_single[j]
        mi_val = mutual_info(rho_ij, rho_i, rho_j, xp=xp)
        return {"Site Pair": f"{i}-{j}", "Mutual Information": mi_val}

    def compute_mi(peps, n_sites, approximate=False, use_gpu=True):
        xp = cp if use_gpu and cupy_available else np
        print(f"Computing MI using {'GPU (CuPy)' if xp is cp else 'CPU (NumPy)'}...")

        try:
            if approximate:
                psi = peps.contract(optimize=cotengra_optimizer)
                psi = xp.array(psi.data) if isinstance(psi, qtn.Tensor) else xp.array(psi)
            else:
                psi = xp.array(peps.to_dense())
            norm = xp.abs(xp.vdot(psi, psi))
            if norm > 0:
                psi /= xp.sqrt(norm)
            else:
                print("Warning: Zero norm in psi, MI may be invalid")
            psi_vec = psi.reshape(-1)
            rho_full = xp.outer(psi_vec, psi_vec.conj())
            trace_rho = xp.trace(rho_full)
            if not xp.isclose(trace_rho, 1.0, rtol=1e-5):
                print(f"Warning: rho_full trace={float(trace_rho)}, normalizing...")
                rho_full /= trace_rho
            rho_full_np = cp.asnumpy(rho_full) if xp is cp and cupy_available else rho_full
            dims = [2] * n_sites
            rho_single = {}
            for i in range(n_sites):
                rho_single[i] = compute_single_site(rho_full_np, dims, i, xp=xp)

            # Prepare MI pair computations
            mi_results = []
            pair_list = list(combinations(range(n_sites), 2))
            total_pairs = len(pair_list)

            if use_gpu and cupy_available and not is_windows and client is not None:
                # Multi-GPU computation with Dask on Linux
                def compute_pair_dask(pair):
                    i, j = pair
                    return compute_mi_pair(peps, n_sites, i, j, rho_full_np, rho_single, approximate, xp=cp)

                scattered_rho_full_np = client.scatter(rho_full_np, broadcast=True)
                scattered_rho_single = client.scatter(rho_single, broadcast=True)

                futures = client.map(compute_pair_dask, pair_list)
                mi_results = client.gather(futures)

                for idx, (i, j) in enumerate(pair_list):
                    mi_val = mi_results[idx]["Mutual Information"]
                    print(f"Pair ({i},{j}): MI={mi_val:.6f}, rho_ij trace computed on GPU")
            else:
                # Single-GPU or CPU computation
                if use_gpu and cupy_available and is_windows:
                    # Single-GPU on Windows
                    for i, j in tqdm(pair_list, total=total_pairs, desc="MI pairs"):
                        result = compute_mi_pair(peps, n_sites, i, j, rho_full_np, rho_single, approximate, xp=xp)
                        print(f"Pair ({i},{j}): MI={result['Mutual Information']:.6f}, rho_ij trace computed on GPU")
                        mi_results.append(result)
                else:
                    # CPU computation with multiprocessing
                    num_cores = mp.cpu_count()
                    print(f"Using {num_cores} CPU cores for parallel computation.")

                    def compute_pair_cpu(pair, rho_full_np=rho_full_np, rho_single=rho_single):
                        i, j = pair
                        result = compute_mi_pair(peps, n_sites, i, j, rho_full_np, rho_single, approximate, xp=np)
                        print(f"Pair ({i},{j}): MI={result['Mutual Information']:.6f}, rho_ij trace computed on CPU")
                        return result

                    with mp.Pool(processes=num_cores) as pool:
                        mi_results = list(tqdm(
                            pool.imap_unordered(
                                partial(compute_pair_cpu, rho_full_np=rho_full_np, rho_single=rho_single),
                                pair_list
                            ),
                            total=total_pairs,
                            desc="MI pairs"
                        ))

            df_mi = pd.DataFrame(mi_results)
            print(f"MI DataFrame: {df_mi.shape}, non-zero MI={len(df_mi[df_mi['Mutual Information'] > 1e-10])}")
            return df_mi
        except Exception as e:
            if use_gpu and cupy_available:
                print(f"CuPy error: {e}. Falling back to CPU (NumPy).")
                return compute_mi(peps, n_sites, approximate, use_gpu=False)
            else:
                raise e
        finally:
            if client is not None:
                client.close()
                cluster.close()

    # --- Time Evolution ---
    print("Starting time evolution...")
    mi_evolution = []
    graphs = []
    for t in range(time_steps):
        print(f"Time step {t+1}/{time_steps} (t={t*dt:.2f})...")
        if t > 0:
            for H_term, (site1, site2) in ham_terms:
                U = qu.expm(-1j * H_term * dt)
                print(f"Applying gate to sites {site1}, {site2}: U shape={U.shape}")
                peps.gate(U, where=(site1, site2), inplace=True)
                peps.compress_all(max_bond=bond_dim)
                print(f"Post-compression bond_dim={peps.max_bond()}")
        G, df_mi = build_graph(peps, Lx, Ly, approximate, use_gpu)
        for i in G.nodes:
            x, y, _ = G.nodes[i]["pos"]
            G.nodes[i]["pos"] = (x, y, t)
        graphs.append(G)
        mi_evolution.append(df_mi)

    # --- Compute Outputs ---
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
        print(f"Step {t}: Graph nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

    # --- Save Outputs ---
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
    run_simulation(Lx=3, Ly=3, use_gpu=True)