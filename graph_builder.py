import quimb as qu
import quimb.tensor as qtn
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.manifold import MDS
from itertools import combinations
from scipy.linalg import eigvalsh
from scipy.sparse import identity, csr_matrix
from tqdm import tqdm
import psutil

# Try to import CuPy for GPU acceleration; fall back to NumPy if unavailable
try:
    import cupy as cp
    print("CuPy is available. GPU acceleration can be used.")
    cupy_available = True
except ImportError:
    print("CuPy is not available. Falling back to NumPy (CPU).")
    cp = np
    cupy_available = False

def entropy(rho, xp=np):
    # Convert CuPy array to NumPy for eigvalsh (runs on CPU)
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

def compute_mi(peps, n_sites, approximate=False, use_gpu=True):
    # Select the array backend based on use_gpu and CuPy availability
    xp = cp if use_gpu and cupy_available else np
    print(f"Computing MI using {'GPU (CuPy)' if xp is cp else 'CPU (NumPy)'}...")
    
    try:
        if approximate:
            psi = peps.contract(optimize='auto-hq')
            # Convert Tensor to array using .data
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
        # Convert rho_full to NumPy array for partial_trace
        rho_full_np = cp.asnumpy(rho_full) if xp is cp and cupy_available else rho_full
        dims = [2] * n_sites
        rho_single = {}
        for i in range(n_sites):
            rho_single[i] = qu.partial_trace(rho_full_np, dims, keep=[i])
            # Convert back to xp (CuPy or NumPy)
            rho_single[i] = xp.array(rho_single[i])
            trace_i = xp.trace(rho_single[i])
            if not xp.isclose(trace_i, 1.0, rtol=1e-5):
                print(f"Warning: rho_single[{i}] trace={float(trace_i)}")
        mi_results = []
        for i, j in tqdm(combinations(range(n_sites), 2), total=len(list(combinations(range(n_sites), 2))), desc="MI pairs"):
            # Convert to NumPy for partial_trace
            rho_ij = qu.partial_trace(rho_full_np, dims, keep=[i, j])
            # Convert back to xp (CuPy or NumPy)
            rho_ij = xp.array(rho_ij)
            rho_i = rho_single[i]
            rho_j = rho_single[j]
            mi_val = mutual_info(rho_ij, rho_i, rho_j, xp=xp)
            print(f"Pair ({i},{j}): MI={mi_val:.6f}, rho_ij trace={float(xp.trace(rho_ij)):.6f}")
            mi_results.append({"Site Pair": f"{i}-{j}", "Mutual Information": mi_val})
        df_mi = pd.DataFrame(mi_results)
        print(f"MI DataFrame: {df_mi.shape}, non-zero MI={len(df_mi[df_mi['Mutual Information'] > 1e-10])}")
        return df_mi
    except Exception as e:
        if use_gpu and cupy_available:
            print(f"CuPy error: {e}. Falling back to CPU (NumPy).")
            return compute_mi(peps, n_sites, approximate, use_gpu=False)
        else:
            raise e

def build_graph(peps, Lx, Ly, approximate, use_gpu=True):
    n_sites = Lx * Ly
    df_mi = compute_mi(peps, n_sites, approximate, use_gpu)
    G = nx.Graph()
    for i in range(n_sites):
        G.add_node(i, pos=(i % Lx, i // Lx, 0))
    for _, row in df_mi.iterrows():
        i, j = map(int, row["Site Pair"].split("-"))
        mi = row["Mutual Information"]
        if mi > 1e-20:
            G.add_edge(i, j, weight=mi, distance=1 / (mi + 1e-20))
            print(f"Added edge {i}-{j} with MI={mi:.6f}")
    print(f"Graph: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
    dist_matrix = np.zeros((n_sites, n_sites))
    for i, j, data in G.edges(data=True):
        dist_matrix[i, j] = dist_matrix[j, i] = data["distance"]
    dist_matrix[np.isinf(dist_matrix) | (dist_matrix == 0)] = 1000.0
    np.fill_diagonal(dist_matrix, 0)
    mds = MDS(n_components=3, dissimilarity="precomputed", random_state=42)
    pos_3d = mds.fit_transform(dist_matrix)
    for i, pos in enumerate(pos_3d):
        G.nodes[i]["pos_3d"] = pos
    return G, df_mi

def build_heisenberg_ham(Lx, Ly, J=1.0, cyclic=False):
    terms = []
    sx = qu.pauli('X')
    sy = qu.pauli('Y')
    sz = qu.pauli('Z')
    
    for i in range(Lx):
        for j in range(Ly):
            if j < Ly - 1 or cyclic:
                H_term = J * (qu.kron(sx, sx) + qu.kron(sy, sy) + qu.kron(sz, sz))
                terms.append((H_term, ((i, j), (i, j+1))))
            if i < Lx - 1 or cyclic:
                H_term = J * (qu.kron(sx, sx) + qu.kron(sy, sy) + qu.kron(sz, sz))
                terms.append((H_term, ((i, j), (i+1, j))))
    return terms

def evolve_peps(Lx, Ly, bond_dim=2, time_steps=5, dt=0.1, hamiltonian="heisenberg", approximate=False, use_gpu=True):
    n_sites = Lx * Ly
    available_memory = psutil.virtual_memory().available / (1024 ** 3)
    required_memory = (2 ** n_sites) ** 2 * 16 / (1024 ** 3)
    if available_memory < required_memory * 1.5:
        approximate = True
        print("Using approximate contraction due to memory constraints.")
    peps = qtn.PEPS.rand(Lx=Lx, Ly=Ly, bond_dim=bond_dim, phys_dim=2, seed=42)
    if hamiltonian == "heisenberg":
        ham = build_heisenberg_ham(Lx, Ly, J=1.0, cyclic=False)
    elif hamiltonian == "ising":
        raise NotImplementedError("Ising Hamiltonian not yet implemented.")
    else:
        raise ValueError("Hamiltonian must be 'heisenberg' or 'ising'.")
    graphs = []
    mi_dfs = []
    for t in tqdm(range(time_steps), desc="Evolving PEPS"):
        if t > 0:
            for H_term, (site1, site2) in ham:
                U = qu.expm(-1j * H_term * dt)
                peps.gate(U, where=(site1, site2), inplace=True)
                peps.compress_all(max_bond=bond_dim)
        G, df_mi = build_graph(peps, Lx, Ly, approximate, use_gpu)
        for i in G.nodes:
            x, y, _ = G.nodes[i]["pos"]
            G.nodes[i]["pos"] = (x, y, t)
        graphs.append(G)
        mi_dfs.append(df_mi)
    return graphs, mi_dfs