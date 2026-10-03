import torch
from torch import nn


class PersistentState(nn.Module):
    def __init__(self, d_model=256, d_state=256):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state

        self.input_projection = nn.Linear(
            d_model,
            d_state,
        )

        self.gate_projection = nn.Linear(
            d_state * 2,
            d_state,
        )

        self.candidate_projection = nn.Linear(
            d_state * 2,
            d_state,
        )

        self.norm = nn.LayerNorm(d_state)

    def initial_state(self, batch_size, device=None):
        return torch.zeros(
            batch_size,
            self.d_state,
            device=device,
        )

    def forward(self, representation, state):
        # Compress sequence representation into a context vector.
        context = representation.mean(dim=1)

        context = self.input_projection(context)

        combined = torch.cat(
            [context, state],
            dim=-1,
        )

        gate = torch.sigmoid(
            self.gate_projection(combined)
        )

        candidate = torch.tanh(
            self.candidate_projection(combined)
        )

        new_state = (
            (1.0 - gate) * state
            + gate * candidate
        )

        new_state = self.norm(new_state)

        return new_state
