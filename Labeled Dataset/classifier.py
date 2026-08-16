import torch
import torch.nn as nn


class SentenceClassifier(nn.Module):
    def __init__(self, embedding_dim=768):
        super().__init__()

        self.classifier =nn.Linear(embedding_dim, 1)

    def forward(self, embeddings):
        logits = self.classifier(embeddings)

        return logits

    def predict_probability(self, embeddings):
        logits = self.forward(embeddings)

        probabilities = torch.sigmoid(logits)

        return probabilities