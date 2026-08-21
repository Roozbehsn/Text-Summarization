# import torch

# def select_top_k(sentences, scores, k=3):

#     if len(sentences) != len(scores):
#         raise ValueError(
#             "Number of sentences and scores must be the same."
#         )

#     if len(sentences) == 0:
#         return []

#     k = min(k, len(sentences))

#     top_indices = torch.argsort(
#         scores,
#         descending=True
#     )[:k]

#     top_indices = top_indices.tolist()

#     # Restore original article order
#     top_indices.sort()

#     selected_sentences = [
#         sentences[i]
#         for i in top_indices
#     ]

#     return selected_sentences


#MMR
import torch
from sklearn.metrics.pairwise import cosine_similarity


def select_mmr(
    sentences,
    scores,
    embeddings,
    k=3,
    lambda_param=0.7
):

    if len(sentences) == 0:
        return []

    if len(sentences) != len(scores):
        raise ValueError(
            "Number of sentences and scores must match."
        )

    if len(sentences) != len(embeddings):
        raise ValueError(
            "Number of sentences and embeddings must match."
        )

    k = min(k, len(sentences))

    # Convert BERT embeddings to numpy
    embeddings = embeddings.detach().cpu().numpy()

    # Calculate sentence-to-sentence similarity
    similarity_matrix = cosine_similarity(
        embeddings
    )

    scores = scores.detach().cpu()

    selected_indices = []


    first_index = torch.argmax(scores).item()

    selected_indices.append(first_index)

    while len(selected_indices) < k:

        best_index = None
        best_mmr_score = float("-inf")

        for i in range(len(sentences)):

            if i in selected_indices:
                continue
            importance = scores[i].item()

            redundancy = max(
                similarity_matrix[i][j]
                for j in selected_indices
            )


            mmr_score = (
                lambda_param * importance
                -
                (1 - lambda_param) * redundancy
            )

            if mmr_score > best_mmr_score:

                best_mmr_score = mmr_score
                best_index = i

        if best_index is None:
            break

        selected_indices.append(best_index)

    selected_indices.sort()

    selected_sentences = [
        sentences[i]
        for i in selected_indices
    ]

    return selected_sentences