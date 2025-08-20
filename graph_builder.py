import networkx as nx
import numpy as np
import quimb as qu
import quimb.tensor as qtn
import itertools
import pandas as pd
from tqdm import tqdm
from sklearn.manifold import MDS
from scipy.linalg import eigvalsh

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    print("CuPy is available. GPU acceleration can be used.")
    cupy_available = True
except ImportError:
    print("CuPy is not available. Falling back to NumPy (CPU).")
    cp = np
    cupy_available = False

def entropy(rho, xp=np):
    # Convert CuPy array to NumPy for eigvalsh if needed
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

def compute_mi(peps, Lx, Ly, approximate=False, use_gpu=True):
    print("Using graph_builder.py version ID: e8f3c2a5-4b9d-4e2f-8a1c-6d7b0e9f3g4h")
    # Select array backend
    xp = cp if use_gpu and cupy_available else np
    print(f"Computing MI using {'GPU (CuPy)' if xp is cp else 'CPU (NumPy)'}...")
    
    try:
        n_sites = Lx * Ly
        # Ensure PEPS is normalized
        peps_norm = peps.norm()
        if peps_norm > 0:
            peps /= peps_norm  # Ensure normalization
        else:
            raise ValueError("PEPS norm is zero, cannot compute MI.")
        mi_results = []
        # Precompute single-site reduced density matrices
        rho_single = {}
        # Define the dimensions for partial trace: each site has dimension 2 (qubit)
        dims = [2] * n_sites
        for i in range(n_sites):
            # Convert linear index to coordinates
            x, y = i % Lx, i // Lx
            site_coord = (x, y)
            print(f"Computing density matrix for site index {i} at coordinates {site_coord}")
            # Compute the reduced density matrix using partial_trace
            # First, convert PEPS to a dense state for partial_trace
            psi = peps.to_dense()
            psi = np.ascontiguousarray(psi, dtype=np.complex128)
            # Normalize the state
            norm = np.abs(np.vdot(psi, psi))
            if norm > 0:
                psi /= np.sqrt(norm)
            # Compute the full density matrix
            psi_vec = psi.reshape(-1)
            rho_full = np.outer(psi_vec, psi_vec.conj())
            trace_rho = np.trace(rho_full)
            if not np.isclose(trace_rho.real, 1.0, rtol=1e-5):
                print(f"Warning: rho_full trace={trace_rho}, normalizing...")
                rho_full /= trace_rho.real
            # Compute single-site reduced density matrix
            keep_indices = [i]  # Keep the i-th site
            rho_i = qu.partial_trace(rho_full, dims, keep=keep_indices)
            rho_i = np.ascontiguousarray(rho_i, dtype=np.complex128)
            # Transfer to GPU if CuPy is used
            rho_i_gpu = xp.asarray(rho_i, dtype=xp.complex128)
            rho_single[i] = rho_i_gpu
            trace_i = xp.trace(rho_single[i])
            # Check the imaginary part of the trace
            imag_part = xp.imag(trace_i)
            if not xp.isclose(imag_part, 0.0, rtol=1e-5):
                print(f"Warning: Imaginary part of trace for rho_single[{i}] (coord {site_coord}) is {float(imag_part)}, expected ~0")
            trace_i_real = xp.real(trace_i)
            if not xp.isclose(trace_i_real, 1.0, rtol=1e-5):
                print(f"Warning: rho_single[{i}] (coord {site_coord}) trace (real part)={float(trace_i_real)}")
                if trace_i_real != 0:
                    rho_single[i] /= trace_i_real
                else:
                    raise ValueError(f"rho_single[{i}] (coord {site_coord}) trace (real part) is zero, cannot normalize.")
        for i, j in tqdm(itertools.combinations(range(n_sites), 2), total=len(list(itertools.combinations(range(n_sites), 2))), desc="MI pairs"):
            # Convert linear indices to coordinates
            x_i, y_i = i % Lx, i // Lx
            x_j, y_j = j % Lx, j // Lx
            site_i = (x_i, y_i)
            site_j = (x_j, y_j)
            print(f"Computing two-site density matrix for sites indices ({i},{j}) at coordinates ({site_i},{site_j})")
            # Compute two-site reduced density matrix
            keep_indices = [i, j]  # Keep the i-th and j-th sites
            rho_ij = qu.partial_trace(rho_full, dims, keep=keep_indices)
            rho_ij = np.ascontiguousarray(rho_ij, dtype=np.complex128)
            # Transfer to GPU if CuPy is used
            rho_ij_gpu = xp.asarray(rho_ij, dtype=xp.complex128)
            rho_i = rho_single[i]
            rho_j = rho_single[j]
            trace_ij = xp.trace(rho_ij_gpu)
            # Check the imaginary part of the trace
            imag_part = xp.imag(trace_ij)
            if not xp.isclose(imag_part, 0.0, rtol=1e-5):
                print(f"Warning: Imaginary part of trace for rho_ij[{i},{j}] (coords {site_i},{site_j}) is {float(imag_part)}, expected ~0")
            trace_ij_real = xp.real(trace_ij)
            if not xp.isclose(trace_ij_real, 1.0, rtol=1e-5):
                print(f"Warning: rho_ij[{i},{j}] (coords {site_i},{site_j}) trace (real part)={float(trace_ij_real)}")
                if trace_ij_real != 0:
                    rho_ij_gpu /= trace_ij_real
                else:
                    raise ValueError(f"rho_ij[{i},{j}] (coords {site_i},{site_j}) trace (real part) is zero, cannot normalize.")
            mi_val = mutual_info(rho_ij_gpu, rho_i, rho_j, xp=xp)
            print(f"Pair ({i},{j}) (coords {site_i},{site_j}): MI={mi_val:.6f}, rho_ij trace (real part)={float(trace_ij_real)}")
            mi_results.append({"Site Pair": f"{i}-{j}", "Mutual Information": mi_val})
        df_mi = pd.DataFrame(mi_results)
        print(f"MI DataFrame: {df_mi.shape}, non-zero MI={len(df_mi[df_mi['Mutual Information'] > 1e-10])}")
        return df_mi
    except Exception as e:
        if use_gpu and cupy_available:
            print(f"CuPy error: {e}. Falling back to CPU (NumPy).")
            return compute_mi(peps, Lx, Ly, approximate, use_gpu=False)
        else:
            raise e

def build_graph(peps, Lx, Ly, approximate=False, use_gpu=True):
    n_sites = Lx * Ly
    df_mi = compute_mi(peps, Lx, Ly, approximate, use_gpu)
    G = nx.Graph()
    for i in range(n_sites):
        G.add_node(i, pos=(i % Lx, i // Lx, 0))
    for _, row in df_mi.iterrows():
        i, j = map(int, row["Site Pair"].split("-"))
        mi = row["Mutual Information"]
        if mi > 1e-10:  # Use a small threshold to filter out numerical noise
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