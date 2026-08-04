import matplotlib.pyplot as plt
import networkx as nx

def visualize_graph(G):
    plt.figure(figsize=(8, 6))
    
    pos = nx.spring_layout(G)
    nx.draw(
        G, pos,
        with_labels=True,
        node_color='lightblue',
        edge_color='gray',
        node_size=2000,
        font_size=10
    )

    edge_labels = nx.get_edge_attributes(G, 'weight')

    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()}
    )

    plt.title("Sentence Similarity Graph (with weights)")
    plt.show()