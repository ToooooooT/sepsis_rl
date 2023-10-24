import torch
import torch.nn as nn
import torch.nn.functional as F

class DuellingMLP(nn.Module):
    def __init__(self, input_size, num_actions, hidden_size=(128, 128)):
        super().__init__()
        self.num_actions = num_actions
        self.layer_size = [input_size] + list(hidden_size)
        self.main = []
        for i in range(len(self.layer_size) - 1):
            self.main.append(nn.Linear(self.layer_size[i], self.layer_size[i + 1]))
            # self.main.append(nn.BatchNorm1d(self.layer_size[i + 1]))
            self.main.append(nn.LeakyReLU())
        self.main = nn.Sequential(*self.main)
        self.adv = nn.Linear(self.layer_size[-1] // 2, self.num_actions)
        self.val = nn.Linear(self.layer_size[-1] // 2, 1)
        # self.adv = torch.empty([25, 64]).normal_(mean=0, std=1)
        # self.val = torch.empty([1, 64]).normal_(mean=0, std=1)
    

    def forward(self, x):
        y = self.main(x)
        # advantage and value streams
        streamA, streamV = torch.split(y, self.layer_size[-1] // 2, dim=1)
        adv = self.adv(streamA)
        val = self.val(streamV)
        # adv = streamA @ self.adv.T
        # val = streamV @ self.val.T
        return val + adv - adv.mean()
