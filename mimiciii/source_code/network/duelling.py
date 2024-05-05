import torch
import torch.nn as nn
import torch.nn.functional as F

class DuellingMLP(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        num_actions: int, 
        hidden_size: tuple = (128, 128)
    ):
        super().__init__()
        self.num_actions = num_actions
        self.layer_size = [input_size] + list(hidden_size)
        self.main = []
        for i in range(len(self.layer_size) - 1):
            self.main.append(nn.Linear(self.layer_size[i], self.layer_size[i + 1]))
            # self.main.append(nn.BatchNorm1d(self.layer_size[i + 1]))
            self.main.append(nn.LeakyReLU())
        self.main = nn.Sequential(*self.main)
        self.adv_shape = self.layer_size[-1] // 2
        self.val_shape = self.layer_size[-1] - self.adv_shape
        self.adv = nn.Linear(self.adv_shape, self.num_actions)
        self.val = nn.Linear(self.val_shape, 1)
        # self.adv = torch.empty([25, 64]).normal_(mean=0, std=1)
        # self.val = torch.empty([1, 64]).normal_(mean=0, std=1)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.main(x)
        # advantage and value streams
        streamA, streamV = torch.split(y, (self.adv_shape, self.val_shape), dim=-1)
        adv = self.adv(streamA)
        val = self.val(streamV)
        # adv = streamA @ self.adv.T
        # val = streamV @ self.val.T
        return val + adv - adv.mean()

    def initialize(self):

        def init_weights(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()

        self.main.apply(init_weights)
        self.adv.reset_parameters()
        self.val.reset_parameters()


class WDQN_DuelingMLP(nn.Module):
    def __init__(self, state_dim: int, n_actions: int):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.fc_val = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.fc_adv = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv(state)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + adv - adv.mean(dim=1, keepdim=True)
