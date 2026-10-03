import torch
from torch import nn


class ReasoningStep(nn.Module):
    def __init__(self, d_state=256):
        super().__init__()

        self.update_network = nn.Sequential(
            nn.Linear(d_state * 2, d_state),
            nn.GELU(),
            nn.Linear(d_state, d_state),
        )

        self.gate_network = nn.Sequential(
            nn.Linear(d_state * 2, d_state),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(d_state)

    def forward(self, state, memory_context):
        combined = torch.cat(
            [state, memory_context],
            dim=-1,
        )

        update = self.update_network(combined)
        gate = self.gate_network(combined)

        new_state = state + gate * update

        return self.norm(new_state)


class ReasoningCore(nn.Module):
    def __init__(
        self,
        d_state=256,
        reasoning_steps=4,
    ):
        super().__init__()

        self.reasoning_steps = reasoning_steps

        # Reuse the same reasoning transformation
        # across iterative cycles.
        self.step = ReasoningStep(d_state)

    def forward(self, state, memory_context):
        intermediate_states = []

        for _ in range(self.reasoning_steps):
            state = self.step(
                state,
                memory_context,
            )

            intermediate_states.append(state)

        return state, intermediate_states
