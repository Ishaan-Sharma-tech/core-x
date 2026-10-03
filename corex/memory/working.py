import torch
from torch import nn


class WorkingMemory(nn.Module):
    def __init__(self, slots=16, d_model=256):
        super().__init__()

        self.slots = slots
        self.d_model = d_model

        self.write_projection = nn.Linear(
            d_model,
            d_model,
        )

        self.write_gate = nn.Linear(
            d_model,
            slots,
        )

        self.read_projection = nn.Linear(
            d_model,
            d_model,
        )

        self.norm = nn.LayerNorm(d_model)

    def initial_memory(self, batch_size, device=None):
        return torch.zeros(
            batch_size,
            self.slots,
            self.d_model,
            device=device,
        )

    def write(self, memory, state):
        content = self.write_projection(state)

        weights = torch.softmax(
            self.write_gate(state),
            dim=-1,
        )

        update = (
            weights.unsqueeze(-1)
            * content.unsqueeze(1)
        )

        new_memory = memory + update

        return self.norm(new_memory)

    def read(self, memory, state):
        query = self.read_projection(state)

        scores = (
            memory
            * query.unsqueeze(1)
        ).sum(dim=-1)

        weights = torch.softmax(
            scores,
            dim=-1,
        )

        retrieved = (
            memory
            * weights.unsqueeze(-1)
        ).sum(dim=1)

        return retrieved
