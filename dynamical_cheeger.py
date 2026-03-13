import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# TYPOGRAPHY (NO LATEX INSTALL REQUIRED)
# ============================================================

mpl.rcParams['text.usetex'] = False
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['figure.titlesize'] = 16


# ============================================================
# PARAMETERS
# ============================================================

alpha = 1.0
epsilon = 0.05
delta = 1.0
beta = 0.1
gamma = 2.5


# ============================================================
# RICCATI SPECTRAL MAP
# ============================================================

def p_lambda(lam):
    term = alpha**2 * (beta + gamma * lam) - delta * (2 * epsilon * lam - epsilon**2 * lam**2)
    disc = (
        delta * (2 * epsilon * lam - epsilon**2 * lam**2)
        - alpha**2 * (beta + gamma * lam)
    )**2 + 4 * alpha**2 * delta * (beta + gamma * lam)
    return term / (2 * alpha**2) + np.sqrt(disc) / (2 * alpha**2)


# ============================================================
# MODIFIED BINARY TREE
# ============================================================

def make_binary_tree():
    """
    Ten-node tree with the last leaf rewired from red parent 4 to blue parent 5.
    This makes the displayed 5-5 red/blue split a genuine cut of the topology.
    """
    G = nx.Graph()
    G.add_nodes_from(range(10))
    G.add_edges_from([
        (0, 1), (0, 2),
        (1, 3), (1, 4),
        (2, 5), (2, 6),
        (3, 7), (3, 8),
        (5, 9),
    ])
    return G


def balanced_binary_tree_pos():
    return {
        0: (0.0, 3.0),
        1: (-2.4, 2.0),
        2: (2.4, 2.0),
        3: (-3.8, 1.0),
        4: (-1.2, 1.0),
        5: (1.2, 1.0),
        6: (3.8, 1.0),
        7: (-4.4, 0.0),
        8: (-3.2, 0.0),
        9: (1.2, 0.0),
    }


# ============================================================
# GRAPH SET
# ============================================================

graphs = {
    "Path": nx.path_graph(10),
    "Barbell": nx.barbell_graph(5, 0),
    "Complete": nx.complete_graph(10),
    "Binary Tree": make_binary_tree(),
    "Star": nx.star_graph(9),
    "Karate": nx.karate_club_graph(),
}


# ============================================================
# TRUE CONDUCTANCE VALUES FOR THE DISPLAYED TOPOLOGIES
# ============================================================

phi = {
    "Path": 1 / 9,
    "Barbell": 1 / 21,
    "Complete": 5 / 9,
    "Binary Tree": 1 / 9,
    "Star": 1.0,
    "Karate": 5 / 39,
}


# ============================================================
# ANALYSIS
# ============================================================

results = []

for name, G in graphs.items():
    L = nx.laplacian_matrix(G).toarray()
    lam2 = np.linalg.eigvalsh(L)[1]
    lam2P = p_lambda(lam2)

    degrees = [d for _, d in G.degree()]
    dmin = min(degrees)
    dmax = max(degrees)

    lower_arg = (dmin / 2) * phi[name]**2
    upper_arg = 2 * dmax * phi[name]

    LB = p_lambda(lower_arg)
    UB = p_lambda(upper_arg)

    results.append({
        "name": name,
        "G": G,
        "lambda2P": lam2P,
        "LB": LB,
        "UB": UB,
    })


# ============================================================
# PRINT RESULTS
# ============================================================

print("\nNumerical Results")
print("=" * 80)
for r in results:
    print(
        f"{r['name']:>12} | "
        f"lambda2(P)={r['lambda2P']:.4f} | "
        f"LB={r['LB']:.4f} | "
        f"UB={r['UB']:.4f}"
    )
print("=" * 80)


# ============================================================
# GRAPH VISUALIZATION
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for ax, r in zip(axes, results):
    name = r["name"]
    G = r["G"]

    if name == "Path":
        pos = {i: (i, 0) for i in G.nodes()}
    elif name == "Binary Tree":
        pos = balanced_binary_tree_pos()
    elif name == "Star":
        pos = nx.shell_layout(G, nlist=[[0], list(range(1, 10))])
    elif name == "Complete":
        pos = nx.circular_layout(G)
    elif name == "Barbell":
        pos = nx.spring_layout(G, seed=42, k=0.4)
    elif name == "Karate":
        pos = nx.spring_layout(G, seed=42, k=0.3)
    else:
        pos = nx.spring_layout(G)

    if name == "Path":
        S = set(range(5))
    elif name == "Barbell":
        S = set(range(5))
    elif name == "Complete":
        S = set(range(5))
    elif name == "Binary Tree":
        S = {1, 3, 4, 7, 8}
    elif name == "Star":
        S = {0, 1, 2, 3, 4}
    elif name == "Karate":
        S = {0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 16, 17, 19, 21}

    Sc = set(G.nodes()) - S
    node_colors = ["#e63946" if v in S else "#457b9d" for v in G.nodes()]

    boundary = []
    internal = []
    for u, v in G.edges():
        if (u in S and v in Sc) or (u in Sc and v in S):
            boundary.append((u, v))
        else:
            internal.append((u, v))

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edgelist=internal,
        width=1.5,
        alpha=0.35,
        edge_color="grey",
    )
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edgelist=boundary,
        width=3,
        edge_color="black",
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=300,
        edgecolors="black",
    )

    ax.set_title(name, fontsize=14)
    ax.axis("off")

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=r'Set $S$',
           markerfacecolor='#e63946', markersize=10, markeredgecolor='black'),
    Line2D([0], [0], marker='o', color='w', label=r'Set $S^c$',
           markerfacecolor='#457b9d', markersize=10, markeredgecolor='black'),
    Line2D([0], [0], color='grey', lw=2,
           label=r'Internal Edges in $\mathrm{Vol}(S)\cup \mathrm{Vol}(S^c)$'),
    Line2D([0], [0], color='black', lw=3,
           label=r'Cut Edges in $\partial S$'),
]

fig.legend(handles=legend_elements, loc="lower center", ncol=4, frameon=False, fontsize=14)
fig.suptitle("Representative Graph Topologies and Sparse Cuts", fontsize=16)
plt.tight_layout(rect=[0, 0.08, 1, 0.95])
plt.savefig("./dynamical_cheeger_graphs.png", dpi=400, bbox_inches="tight")
plt.close(fig)


# ============================================================
# CHEEGER BOUND PLOT
# ============================================================

names = [r["name"] for r in results]
LB = [r["LB"] for r in results]
UB = [r["UB"] for r in results]
val = [r["lambda2P"] for r in results]
x = np.arange(len(names))

fig2 = plt.figure(figsize=(11, 5))
for i in range(len(x)):
    plt.vlines(x[i], LB[i], UB[i], color="green", linewidth=3)
    plt.hlines(LB[i], x[i] - 0.25, x[i] + 0.25, color="#1f77b4", linewidth=3)
    plt.hlines(UB[i], x[i] - 0.25, x[i] + 0.25, color="#d62728", linewidth=3)

plt.scatter(x, val, color="black", s=130, zorder=3)
plt.xticks(x, names, fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel("Value", fontsize=14)
plt.title("Dynamical Cheeger Bounds vs Riccati Spectral Gap", fontsize=14)
plt.grid(alpha=0.25)

legend_elements = [
    Line2D([0], [0], color="#1f77b4", lw=3, label="Conductance Based Lower Bound"),
    Line2D([0], [0], color="#d62728", lw=3, label="Conductance Based Upper Bound"),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
           markersize=10, label=r"True Value: $\lambda_2(P)$"),
]

plt.legend(handles=legend_elements, fontsize=14, loc="upper left", bbox_to_anchor=(0.02, 0.98), borderaxespad=0.2)
plt.tight_layout()
plt.savefig("./dynamical_cheeger_bounds.png", dpi=400, bbox_inches="tight")
plt.close(fig2)
