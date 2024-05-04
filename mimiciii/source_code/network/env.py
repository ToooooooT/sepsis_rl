import torch
import torch.nn as nn

class EnvMLP(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.layer_size = [state_dim + action_dim] + list(hidden_size) + [state_dim]
        self.main = []
        for i in range(len(self.layer_size) - 2):
            self.main.append(nn.Linear(self.layer_size[i], self.layer_size[i + 1]))
            self.main.append(nn.BatchNorm1d(self.layer_size[i + 1]))

        self.main = nn.Sequential(*self.main)
        self.output = nn.Linear(self.layer_size[-2], self.layer_size[-1])

    def forward(self, x):
        y = self.main(x)
        return self.output(y)