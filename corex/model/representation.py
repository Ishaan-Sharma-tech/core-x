import torch
from torch import nn


class RepresentationEngine(nn.Module):
    def __init__(self, vocab_size=64, d_model=256):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embedding = nn.Embedding(
            vocab_size,
            d_model,
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.norm(x)
        return x
