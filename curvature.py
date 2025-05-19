import networkx as nx

def ricci_curvature(G, i, j):
    if not G.has_edge(i, j):
        return 0.0
    mi = G[i][j]["weight"]
    if mi < 1e-15:
        return 0.0
    curvature = -mi  # Negative MI for AdS-like geometry
    return curvature

def compute_curvature(G):
    curv_t = {(i, j): ricci_curvature(G, i, j) for i, j in G.edges()}
    return curv_t