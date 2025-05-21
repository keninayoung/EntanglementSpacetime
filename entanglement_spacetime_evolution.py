# EntanglementSpacetime Simulation Framework
# Author(s): Kenneth Young, PhD
#            Paul Bransford
# Created: May 2025
#
# Purpose:
# --------
# This program simulates the emergence of spacetime from quantum entanglement using
# time-evolved tensor networks. It employs a Projected Entangled Pair States (PEPS)
# tensor network evolved under a Heisenberg Hamiltonian to construct an entanglement
# graph, where spacetime distances are defined as d(i, j) ~ -log I(i:j), with I(i:j)
# being the mutual information between quantum sites. The framework computes discrete
# curvature, approximates the Einstein tensor, and analyzes holographic entropy and
# black hole dynamics, providing a computational approach to study quantum gravity and
# holography.
#
# This code supports both single-GPU (Windows/Linux) and multi-GPU (Linux only)
# parallelization, with tensor contraction optimization using Optuna when KaHyPar is
# unavailable.
#
# Publication:
# ------------
# For methodology and results, see:
# "Entanglement-Driven Emergent Spacetime with Time-Evolved Tensor Networks"
# by Kenneth Young, PhD (2025).
#
# Usage:
# ------
# Run a simulation with the following command:
#     python3 -m emergent_spacetime.cli --Lx 3 --Ly 3 --steps 5 --hamiltonian heisenberg --use_gpu True
#
# Requirements:
# -------------
# - Python 3.9+
# - CUDA 12.6 (for GPU support)
# - Minimum 16 GB RAM for 4x4 grids
# - Linux for multi-GPU parallelization (Windows supports single-GPU or CPU)
#
# See README.md for detailed installation and usage instructions.

# Import standard libraries for data handling, numerical operations, and system checks
import pandas as pd  # For DataFrame operations to store and save simulation results
import numpy as np   # For numerical operations (used as fallback if GPU is unavailable)
import os            # For file and directory operations (e.g., saving outputs)
import platform      # To determine the operating system (Windows/Linux)
import quimb as qu   # Quimb library for quantum tensor network operations
import quimb.tensor as qtn  # Quimb's tensor module for PEPS and tensor operations
from tqdm import tqdm  # For progress bars during long computations
import psutil         # To check system memory availability
from itertools import combinations  # To generate pairs of sites for mutual information
from scipy.sparse import identity, csr_matrix  # For sparse matrix operations
from scipy.linalg import eigvalsh  # To compute eigenvalues for entropy calculations
import multiprocessing as mp  # For CPU parallelization
from functools import partial  # To create partial functions for parallel processing

# Check if KaHyPar is available for tensor contraction optimization
try:
    import kahypar
    kahypar_available = True
    print("KaHyPar is available. Using default HyperOptimizer for cotengra.")
except ImportError:
    kahypar_available = False
    print("KaHyPar is not available.")

# Check if Optuna is available as an alternative optimizer for tensor contractions
try:
    import optuna
    optuna_available = True
    print("Optuna is available. Using Optuna-based HyperOptimizer for cotengra.")
except ImportError:
    optuna_available = False
    print("Optuna is not available. Falling back to greedy method for cotengra optimization.")

# Import cotengra to set the optimizer for tensor contractions
import cotengra

# Set the cotengra optimizer based on availability of KaHyPar or Optuna
if kahypar_available:
    cotengra_optimizer = cotengra.HyperOptimizer()  # Default optimizer if KaHyPar is available
elif optuna_available:
    cotengra_optimizer = cotengra.HyperOptimizer(methods=['optuna'])  # Use Optuna if available
else:
    cotengra_optimizer = 'greedy'  # Fallback to greedy method if no optimizers are available

# Import Dask for CPU parallelization (used later in compute_mi function)
import dask

# Import CuPy for GPU support, with fallback to NumPy if unavailable
try:
    import cupy as cp
    print("CuPy is available. GPU acceleration can be used if enabled.")
    cupy_available = True
except ImportError:
    print("CuPy is not available. Falling back to NumPy (CPU).")
    cp = np  # Set cp to np as a fallback
    cupy_available = False

# Import Dask-CUDA only on Linux for multi-GPU support
is_windows = platform.system() == "Windows"
if not is_windows:
    try:
        from dask_cuda import LocalCUDACluster
        from dask.distributed import Client
        print("Dask-CUDA is available on Linux. Multi-GPU acceleration can be used if enabled.")
        dask_cuda_available = True
    except ImportError:
        print("Dask-CUDA is not available. Falling back to single-GPU, Multi-CPU, or single-CPU.")
        dask_cuda_available = False
else:
    print("Dask-CUDA is not supported on Windows. Using single-GPU, Multi-CPU, or single-CPU.")
    dask_cuda_available = False

# Import project-specific modules for simulation tasks
from graph_builder import build_graph, build_heisenberg_ham  # For building entanglement graph and Hamiltonian
from curvature import compute_curvature  # For computing discrete curvature
from einstein_tensor import compute_einstein_tensor  # For approximating the Einstein tensor
from entropy import compute_entropy  # For computing entanglement entropy
from hawking_radiation import compute_hawking_radiation  # For computing mutual information across a horizon
from visualization import save_entanglement_graph  # For saving interactive visualizations

def run_simulation(Lx=3, Ly=3, bond_dim=2, time_steps=5, dt=0.1, output_dir="spacetime_outputs", approximate=False, use_gpu=True):
    # Default cores to 1
    num_cores = 1

    # Calculate total number of quantum sites in the lattice (Lx * Ly)
    n_sites = Lx * Ly

    # Initialize Dask client for multi-GPU computation (Linux only) if GPU is requested
    client = None
    computation_mode = None  # Track the computation mode: "multi-gpu", "single-gpu", "multi-cpu", "single-cpu"

    # Estimate memory requirements for the simulation (used for both multi-GPU and single-GPU checks)
    available_memory = psutil.virtual_memory().available / (1024 ** 3)  # Convert to GB
    required_memory = (2 ** n_sites) ** 2 * 16 / (1024 ** 3)  # Approximate memory for density matrix in GB
    print(f"Available memory: {available_memory:.2f} GB")
    print(f"Estimated memory for density matrix: {required_memory:.2f} GB")

    # Step 1: Attempt multi-GPU computation (highest priority if conditions are met)
    if use_gpu and cupy_available and not is_windows and dask_cuda_available:
        if available_memory >= required_memory * 1.5:
            try:
                # Create a Dask CUDA cluster for multi-GPU parallelization
                cluster = LocalCUDACluster()
                client = Client(cluster)
                computation_mode = "multi-gpu"
                print("Dask CUDA Cluster initialized for multi-GPU computation.")
            except Exception as e:
                # If multi-GPU setup fails (e.g., due to configuration issues), log the error
                print(f"Failed to initialize Dask CUDA Cluster: {e}. Falling back to single-GPU computation if possible.")
                client = None
        else:
            print("Insufficient memory for multi-GPU computation. Falling back to single-GPU computation if possible.")
            client = None

    # Step 2: If multi-GPU isn't used, attempt single-GPU computation
    if client is None and use_gpu and cupy_available:
        if available_memory >= required_memory * 1.5:
            computation_mode = "single-gpu"
            print("Using single-GPU computation with CuPy.")
        else:
            print("Insufficient memory for single-GPU computation. Falling back to CPU-based computation.")
            # Step 3: Check for multi-CPU availability before falling back to single-CPU
            num_cores = mp.cpu_count()
            print(f"Number of CPU cores available: {num_cores}")
            if num_cores > 1:
                # If multiple CPU cores are available, prefer multi-CPU parallelization
                use_gpu = False
                computation_mode = "multi-cpu"
                print("Multiple CPU cores detected. Falling back to multi-CPU parallel computation.")
            else:
                # If only one CPU core is available, fall back to single-CPU
                use_gpu = False
                computation_mode = "single-cpu"
                print("Only one CPU core available. Falling back to single-CPU computation.")

    # Step 3: If computation mode hasn't been set (neither multi-GPU nor single-GPU), determine CPU-based computation mode
    if computation_mode is None:
        # Check the number of CPU cores available for parallel computation
        num_cores = mp.cpu_count()
        print(f"Number of CPU cores available: {num_cores}")
        if num_cores > 1:
            # If multiple CPU cores are available, prefer multi-CPU parallelization
            use_gpu = False
            computation_mode = "multi-cpu"
            print("Multiple CPU cores detected. Using multi-CPU parallel computation.")
        else:
            # If only one CPU core is available, use single-CPU
            use_gpu = False
            computation_mode = "single-cpu"
            print("Only one CPU core available. Using single-CPU computation.")

    # If multi-GPU setup succeeded, confirm the computation mode
    if client is not None:
        computation_mode = "multi-gpu"


    # Dispaly computing mode
    print(f"Computation mode: {computation_mode}")

    # Check memory requirements to determine if approximate contraction is needed
    if approximate or available_memory < required_memory * 1.5 or n_sites >= 16:
        # Use approximate contraction if explicitly requested, memory is insufficient, or lattice is large
        approximate = True
        print("Using approximate contraction due to memory constraints.")

    # Define the Heisenberg Hamiltonian for the PEPS evolution
    print("Defining Heisenberg Hamiltonian...")
    ham_terms = build_heisenberg_ham(Lx, Ly, J=1.0, cyclic=False)

    # Initialize the PEPS tensor network with random states
    print("Building initial PEPS...")
    peps = qtn.PEPS.rand(Lx=Lx, Ly=Ly, bond_dim=bond_dim, phys_dim=2, seed=42)

    # Define helper functions for entropy and mutual information (MI) calculations
    def entropy(rho, xp=np):
        # Compute the von Neumann entropy of a density matrix: S = -Tr(rho log rho)
        # rho: density matrix
        # xp: numpy or cupy module for computation
        if xp is cp and cupy_available:
            try:
                # Convert CuPy array to NumPy for eigenvalue computation
                rho_np = cp.asnumpy(rho)
            except Exception as e:
                print(f"CuPy error in entropy: {e}. Falling back to NumPy.")
                xp = np
                rho_np = rho
        else:
            rho_np = rho
        # Compute eigenvalues, filter out near-zero values to avoid log(0)
        vals = eigvalsh(rho_np)
        vals = vals[vals > 1e-20]
        return -xp.sum(vals * xp.log(vals)) if len(vals) > 0 else 0.0

    def mutual_info(rho_ij, rho_i, rho_j, xp=np):
        # Compute mutual information: I(i:j) = S(rho_i) + S(rho_j) - S(rho_ij)
        # rho_ij: joint density matrix for sites i and j
        # rho_i, rho_j: reduced density matrices for sites i and j
        mi = entropy(rho_i, xp) + entropy(rho_j, xp) - entropy(rho_ij, xp)
        return max(0.0, float(mi))

    def compute_single_site(rho_full_np, dims, i, xp=np):
        # Compute the reduced density matrix for a single site by tracing out others
        # rho_full_np: full density matrix
        # dims: dimensions of the full Hilbert space (list of 2s for qubits)
        # i: site index to keep
        rho_single = qu.partial_trace(rho_full_np, dims, keep=[i])
        rho_single = xp.array(rho_single)
        # Verify the trace of the reduced density matrix is 1 (within tolerance)
        trace_i = xp.trace(rho_single)
        if not xp.isclose(trace_i, 1.0, rtol=1e-5):
            print(f"Warning: rho_single[{i}] trace={float(trace_i)}")
        return rho_single

    def compute_mi_pair(peps, n_sites, i, j, rho_full_np, rho_single, dims, approximate=False, xp=np):
        # Compute mutual information for a pair of sites (i, j)
        # peps: PEPS tensor network
        # n_sites: total number of sites
        # i, j: site indices
        # rho_full_np: full density matrix
        # rho_single: dictionary of single-site reduced density matrices
        # dims: dimensions of the full Hilbert space (list of 2s for qubits)
        # approximate: whether to use approximate contraction
        # xp: numpy or cupy module for computation
        rho_ij = qu.partial_trace(rho_full_np, dims, keep=[i, j])
        rho_ij = xp.array(rho_ij)
        rho_i = rho_single[i]
        rho_j = rho_single[j]
        mi_val = mutual_info(rho_ij, rho_i, rho_j, xp=xp)
        return {"Site Pair": f"{i}-{j}", "Mutual Information": mi_val}

    def compute_mi(peps, n_sites, approximate=False, use_gpu=True):
        # Compute mutual information (MI) between all pairs of sites in the PEPS
        # peps: PEPS tensor network
        # n_sites: total number of sites
        # approximate: whether to use approximate contraction
        # use_gpu: whether to attempt GPU computation
        xp = cp if use_gpu and cupy_available else np
        print(f"Computing MI using {'GPU (CuPy)' if xp is cp else 'CPU (NumPy)'}...")

        try:
            # Convert PEPS to a dense state vector (psi) for density matrix computation
            if approximate:
                psi = peps.contract(optimize=cotengra_optimizer)
                psi = xp.array(psi.data) if isinstance(psi, qtn.Tensor) else xp.array(psi)
            else:
                psi = xp.array(peps.to_dense())
            # Normalize the state vector
            norm = xp.abs(xp.vdot(psi, psi))
            if norm > 0:
                psi /= xp.sqrt(norm)
            else:
                print("Warning: Zero norm in psi, MI may be invalid")
            psi_vec = psi.reshape(-1)
            # Compute the full density matrix rho = |psi><psi|
            rho_full = xp.outer(psi_vec, psi_vec.conj())
            trace_rho = xp.trace(rho_full)
            # Ensure the density matrix is properly normalized (trace = 1)
            if not xp.isclose(trace_rho, 1.0, rtol=1e-5):
                print(f"Warning: rho_full trace={float(trace_rho)}, normalizing...")
                rho_full /= trace_rho
            # Convert to NumPy for partial tracing (if using GPU)
            rho_full_np = cp.asnumpy(rho_full) if xp is cp and cupy_available else rho_full
            dims = [2] * n_sites  # Hilbert space dimensions (2 for qubits)
            rho_single = {}
            # Compute reduced density matrices for each site
            for i in range(n_sites):
                rho_single[i] = compute_single_site(rho_full_np, dims, i, xp=xp)

            # Prepare MI pair computations for all pairs of sites
            mi_results = []
            pair_list = list(combinations(range(n_sites), 2))
            total_pairs = len(pair_list)

            if computation_mode == "multi-gpu":
                # Multi-GPU computation with Dask on Linux
                def compute_pair_dask(pair):
                    i, j = pair
                    return compute_mi_pair(peps, n_sites, i, j, rho_full_np, rho_single, dims, approximate, xp=cp)

                # Scatter data to all GPU workers
                scattered_rho_full_np = client.scatter(rho_full_np, broadcast=True)
                scattered_rho_single = client.scatter(rho_single, broadcast=True)

                # Compute MI pairs in parallel across GPUs
                futures = client.map(compute_pair_dask, pair_list)
                mi_results = client.gather(futures)

                for idx, (i, j) in enumerate(pair_list):
                    mi_val = mi_results[idx]["Mutual Information"]
                    print(f"Pair ({i},{j}): MI={mi_val:.6f}, rho_ij trace computed on GPU")
            elif computation_mode == "single-gpu":
                # Single-GPU computation
                for i, j in tqdm(pair_list, total=total_pairs, desc="MI pairs"):
                    result = compute_mi_pair(peps, n_sites, i, j, rho_full_np, rho_single, dims, approximate, xp=xp)
                    print(f"Pair ({i},{j}): MI={result['Mutual Information']:.6f}, rho_ij trace computed on GPU")
                    mi_results.append(result)
            else:
                # CPU computation (multi-CPU or single-CPU)
                def compute_pair_cpu(pair, rho_full_np=rho_full_np, rho_single=rho_single, dims=dims):
                    i, j = pair
                    result = compute_mi_pair(peps, n_sites, i, j, rho_full_np, rho_single, dims, approximate, xp=np)
                    print(f"Pair ({i},{j}): MI={result['Mutual Information']:.6f}, rho_ij trace computed on CPU")
                    return result

                if computation_mode == "multi-cpu":
                    # Multi-CPU computation using multiprocessing
                    num_cores = mp.cpu_count()
                    with mp.Pool(processes=num_cores) as pool:
                        mi_results = list(tqdm(
                            pool.imap_unordered(
                                partial(compute_pair_cpu, rho_full_np=rho_full_np, rho_single=rho_single, dims=dims),
                                pair_list
                            ),
                            total=total_pairs,
                            desc="MI pairs"
                        ))
                else:
                    # Single-CPU computation
                    for i, j in tqdm(pair_list, total=total_pairs, desc="MI pairs"):
                        result = compute_pair_cpu((i, j), rho_full_np, rho_single, dims)
                        mi_results.append(result)

            # Compile MI results into a DataFrame
            df_mi = pd.DataFrame(mi_results)
            print(f"MI DataFrame: {df_mi.shape}, non-zero MI={len(df_mi[df_mi['Mutual Information'] > 1e-10])}")
            return df_mi
        except Exception as e:
            if use_gpu and cupy_available:
                # If GPU computation fails, fall back to CPU
                print(f"CuPy error: {e}. Falling back to CPU (NumPy).")

                #Fall back to CPU
                if num_cores > 1:
                    # If multiple CPU cores are available, prefer multi-CPU parallelization
                    use_gpu = False
                    computation_mode = "multi-cpu"
                    print("Falling back to multi-CPU parallel computation.")
                else:
                    # If only one CPU core is available, use single-CPU
                    use_gpu = False
                    computation_mode = "single-cpu"
                    print("Falling back to single-CPU computation.")
                # Print compute node
                print(f"Compute mode: {computation_mode}")

                return compute_mi(peps, n_sites, approximate, use_gpu=False)
            
        finally:
            # Clean up Dask client and cluster if they were initialized
            if client is not None:
                client.close()
                cluster.close()

    # Perform time evolution of the PEPS over the specified number of steps
    print("Starting time evolution...")
    mi_evolution = []  # Store mutual information for each time step
    graphs = []        # Store entanglement graphs for each time step
    for t in range(time_steps):
        print(f"Time step {t+1}/{time_steps} (t={t*dt:.2f})...")
        if t > 0:
            # Apply unitary evolution operators (gates) for each Hamiltonian term
            for H_term, (site1, site2) in ham_terms:
                U = qu.expm(-1j * H_term * dt)  # Compute the evolution operator U = exp(-i H dt)
                print(f"Applying gate to sites {site1}, {site2}: U shape={U.shape}")
                peps.gate(U, where=(site1, site2), inplace=True)
                # Compress the PEPS to maintain the bond dimension
                peps.compress_all(max_bond=bond_dim)
                print(f"Post-compression bond_dim={peps.max_bond()}")
        # Build the entanglement graph and compute MI for this time step
        G, df_mi = build_graph(peps, Lx, Ly, approximate, use_gpu)
        # Update the time coordinate in the graph nodes
        for i in G.nodes:
            x, y, _ = G.nodes[i]["pos"]
            G.nodes[i]["pos"] = (x, y, t)
        graphs.append(G)
        mi_evolution.append(df_mi)

    # Compute simulation outputs: curvature, entropy, Hawking radiation, Einstein tensor
    print("Computing outputs...")
    curvature_evolution = []  # Store curvature over time
    hawking_mi = []           # Store mutual information across the horizon (Page curve)
    einstein_approx = []      # Store Einstein tensor approximations
    entropies = []            # Store entanglement entropy over time
    for t, (G, df_mi) in enumerate(zip(graphs, mi_evolution)):
        curv_t = compute_curvature(G)  # Compute discrete curvature
        curvature_evolution.append(curv_t)
        entropies.append(compute_entropy(df_mi, n_sites))  # Compute entanglement entropy
        hawking_mi.append(compute_hawking_radiation(df_mi, Lx, Ly, n_sites))  # Compute Page curve
        einstein_approx.append(compute_einstein_tensor(G, curv_t))  # Compute Einstein tensor
        save_entanglement_graph(G, t, dt, output_dir)  # Save interactive graph visualization
        print(f"Step {t}: Graph nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

    # Save all computed outputs to CSV files
    print("Saving outputs...")
    # Save curvature evolution as a DataFrame with site pairs as rows and time steps as columns
    curv_df = pd.DataFrame(
        {f"Step {t}": {f"{i}-{j}": v for (i, j), v in curv_t.items()}
         for t, curv_t in enumerate(curvature_evolution)}
    )
    curv_df.to_csv(os.path.join(output_dir, "curvature_evolution.csv"))
    # Save mutual information across the horizon (Page curve)
    hawking_df = pd.DataFrame({"Step": range(time_steps), "MI Across Horizon": hawking_mi})
    hawking_df.to_csv(os.path.join(output_dir, "hawking_radiation.csv"))
    # Save Einstein tensor approximations
    einstein_df = pd.DataFrame(
        {f"Step {t}": einstein_t for t, einstein_t in enumerate(einstein_approx)}
    )
    einstein_df.to_csv(os.path.join(output_dir, "einstein_tensor.csv"))
    # Save entanglement entropy over time
    entropy_df = pd.DataFrame({"Step": range(time_steps), "Entropy": entropies})
    entropy_df.to_csv(os.path.join(output_dir, "entropy.csv"))
    print(f"Outputs saved in {output_dir}")

if __name__ == "__main__":
    # Run the simulation with default parameters (3x3 grid, GPU enabled)
    run_simulation(Lx=3, Ly=3, use_gpu=True)