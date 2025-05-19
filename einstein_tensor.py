def compute_einstein_tensor(G, curv_t):
       einstein_t = {}
       for i in range(G.number_of_nodes()):
           local_curv = sum(curv_t.get((i, j), 0) for j in G.neighbors(i))
           einstein_t[i] = local_curv / max(1, G.degree(i))
       return einstein_t