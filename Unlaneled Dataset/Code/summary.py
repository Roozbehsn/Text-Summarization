def generate_summary(sentences, selected_indices, num_sentences=5):
    summary = " ".join(
        sentences[index] for index in selected_indices
    )

    return summary