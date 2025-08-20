import networkx as nx

def ricci_curvature(G, i, j):
    """
    Compute Ricci curvature for a single edge, matching GitHub's implementation.

    Parameters:
    - G: NetworkX graph with MI as edge weights.
    - i, j: Nodes defining the edge.

    Returns:
    - curvature: Negative MI for the edge (approx: -0.002680 to 0.0).
    """
    if not G.has_edge(i, j):
        return 0.0
    mi = G[i][j]["weight"]
    if mi < 1e-15:
        return 0.0
    curvature = -mi  # Negative MI for AdS-like geometry
    return curvature

def compute_curvature(G):
    """
    Compute discrete Ricci curvature for all edges, matching GitHub's implementation.

    Parameters:
    - G: NetworkX graph with MI as edge weights.

    Returns:
    - curv_t: Dictionary of (i,j) edge tuples to curvature values (approx: -0.002680 to 0.0).
    """
    curv_t = {(i, j): ricci_curvature(G, i, j) for i, j in G.edges()}
    # Ensure symmetric entries
    for i, j in list(curv_t.keys()):
        curv_t[(j, i)] = curv_t[(i, j)]
    print(f"Computed curvature: min={min(curv_t.values()):.6f}, max={max(curv_t.values()):.6f}")
    return curv_t