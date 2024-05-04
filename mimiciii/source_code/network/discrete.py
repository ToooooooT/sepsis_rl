import torch.nn as nn
import torch
from typing import Tuple

class PolicyMLP(nn.Module):
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_size: Tuple = (128, 128)
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.layer_size = [state_dim] + list(hidden_size) + [action_dim]
        self.main = []
        for i in range(len(self.layer_size) - 2):
            self.main.append(nn.Linear(self.layer_size[i], self.layer_size[i + 1]))
            self.main.append(nn.LeakyReLU())
        self.main = nn.Sequential(*self.main)
        self.output = nn.Linear(self.layer_size[-2], self.layer_size[-1])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.main(x)
        logits = self.output(y)
        return logits
