import matplotlib.pyplot as plt
import networkx as nx

import matplotlib.pyplot as plt
import networkx as nx


def visualize_graph(G, sentence_scores=None):
    plt.figure(figsize=(14, 10))

    
    pos = nx.spring_layout(
        G,
        k=4,        
        iterations=500, 
        seed=42
    )

    # Node sizes
    if sentence_scores is not None:
        node_sizes = [
            600 + sentence_scores.get(node, 0) * 1200
            for node in G.nodes()
        ]
    else:
        node_sizes = 800

    # Edge widths
    edge_widths = [
        1 + G[u][v]["weight"] * 8
        for u, v in G.edges()
    ]

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color="lightblue",
        node_size=node_sizes,
        edgecolors="black",
        linewidths=1
    )

    nx.draw_networkx_edges(
        G,
        pos,
        width=edge_widths,
        edge_color="gray",
        alpha=0.7
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=11,
        font_weight="bold"
    )

    edge_labels = {
        (u, v): f"{d['weight']:.2f}"
        for u, v, d in G.edges(data=True)
    }

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=8,
        rotate=False,
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.8
        )
    )

    plt.title("Sentence Similarity Graph", fontsize=18)
    plt.axis("off")
    plt.tight_layout()
    plt.show()



def visualize_similarity_matrix(similarity_matrix):
    plt.figure(figsize=(8, 6))

    plt.imshow(
        similarity_matrix,
        interpolation="nearest",
        aspect="auto"
    )

    plt.colorbar(label="Cosine Similarity")

    plt.title("Sentence Similarity Matrix")
    plt.xlabel("Sentence Index")
    plt.ylabel("Sentence Index")

    plt.xticks(range(len(similarity_matrix)))
    plt.yticks(range(len(similarity_matrix)))

    plt.tight_layout()
    plt.show()