import networkx as nx

def textrank(output, damping=0.85, max_iterations=100, tolerance=1e-6):
    scores = {node: 1.0 for node in output.nodes()}

    num_nodes = len(scores)

    if num_nodes == 0:
        return {}

    for _ in range(max_iterations):
        previous_scores = scores.copy()

        for node in output.nodes():
            rank = 0.0

            for neighbor in output.neighbors(node):
                edge_weight = output[node][neighbor]["weight"]

                total_weight = sum(
                    output[neighbor][n]["weight"]
                    for n in output.neighbors(neighbor)
                )

                if total_weight != 0:
                    rank += (
                        previous_scores[neighbor]
                        * edge_weight
                        / total_weight
                    )

            scores[node] = (1 - damping) + damping * rank

        difference = sum(
            abs(scores[node] - previous_scores[node])
            for node in output.nodes()
        )

        if difference < tolerance:
            break

    return scores