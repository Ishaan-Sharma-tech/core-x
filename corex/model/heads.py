from torch import nn


class PredictionHead(nn.Module):
    def __init__(self, d_state=256, num_classes=2):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(d_state, d_state),
            nn.GELU(),
            nn.LayerNorm(d_state),
            nn.Linear(d_state, num_classes),
        )

    def forward(self, state):
        return self.network(state)
