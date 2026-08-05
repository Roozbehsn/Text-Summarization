def remove_redundancy(sentence_scores,
                       similarity_matrix,
                       num_sentences=5,
                       similarity_threshold=0.7):

    ranked_sentences = sorted(
        sentence_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    selected_indices = []

    for index, score in ranked_sentences:

        redundant = False

        for selected in selected_indices:
            if similarity_matrix[index][selected] >= similarity_threshold:
                redundant = True
                break

        if not redundant:
            selected_indices.append(index)

        if len(selected_indices) == num_sentences:
            break

    selected_indices.sort()

    return selected_indices