import plotly.graph_objects as go
import os

def save_entanglement_graph(G, t, dt, output_dir):
    edge_x, edge_y, edge_z = [], [], []
    edge_weights = []
    for i, j in G.edges():
        x0, y0, z0 = G.nodes[i]["pos_3d"]
        x1, y1, z1 = G.nodes[j]["pos_3d"]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
        edge_weights.append(G[i][j]["weight"])
    node_x, node_y, node_z = zip(*[G.nodes[i]["pos_3d"] for i in G.nodes()])
    max_weight = max(edge_weights, default=1.0)
    norm_weights = [w / max_weight for w in edge_weights] if max_weight > 0 else edge_weights
    print(f"Step {t}: Edges={len(edge_x)//3}, Nodes={len(node_x)}, Max MI={max_weight:.6f}, Weight Range=[{min(edge_weights, default=0):.6f}, {max_weight:.6f}]")
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z, mode="lines",
        line=dict(width=4, color=norm_weights, colorscale='Viridis', showscale=True)
    )
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z, mode="markers+text",
        marker=dict(size=6), text=list(range(G.number_of_nodes()))
    )
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=f"3D Entanglement Graph (t={t*dt:.2f})",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        showlegend=False
    )
    try:
        fig.write_html(os.path.join(output_dir, f"entanglement_graph_t{t}.html"))
        print(f"Saved entanglement_graph_t{t}.html")
    except Exception as e:
        print(f"Error saving entanglement_graph_t{t}.html: {e}")