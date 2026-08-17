import torch

def select_top_k(sentences, scores, k=3):
  top_indices = torch.argsort(scores, descending=True)[:k]
  top_indices = top_indices.tolist()
  top_indices.sort()

  selected_sentences =[
        sentences[i]
        for i in top_indices
    ]
  return selected_sentences