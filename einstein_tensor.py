def compute_einstein_tensor(G, curv_t):
    """
    Compute Einstein tensor components for each node, matching GitHub's implementation.

    Parameters:
    - G: NetworkX graph with MI as edge weights.
    - curv_t: Dictionary of edge tuples to curvature values.

    Returns:
    - einstein_t: List of tensor components (~-0.002 to 0.0).
    """
    einstein_t = []
    for i in range(G.number_of_nodes()):
        local_curv = sum(curv_t.get((i, j), 0) for j in G.neighbors(i))
        einstein_val = local_curv / max(1, G.degree(i))
        einstein_t.append(einstein_val)
    print(f"Computed Einstein tensor: min={min(einstein_t):.6f}, max={max(einstein_t):.6f}")
    return einstein_t