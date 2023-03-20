import torch
import torch.nn as nn
import torch.nn.functional as F

class D3QN(nn.Module):
    def __init__(self, input_size, num_actions):
        self.hidden1_size = 128
        self.hidden2_size = 128
        self.num_actions = num_actions
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(input_size, self.hidden1_size),
            nn.BatchNorm1d(self.hidden1_size),
            nn.LeakyReLU()
        )
        self.l2 = nn.Sequential(
            nn.Linear(self.hidden1_size, self.hidden2_size),
            nn.BatchNorm1d(self.hidden2_size),
            nn.LeakyReLU()
        )
        self.adv = nn.Linear(self.hidden2_size // 2, self.num_actions)
        self.val = nn.Linear(self.hidden2_size // 2, 1)
        # self.adv = torch.empty([25, 64]).normal_(mean=0, std=1)
        # self.val = torch.empty([1, 64]).normal_(mean=0, std=1)
    

    def forward(self, x):
        y = self.l1(x)
        y = self.l2(y)
        # advantage and value streams
        streamA, streamV = torch.split(y, self.hidden2_size // 2, dim=1)
        adv = self.adv(streamA)
        val = self.val(streamV)
        # adv = self.adv @ streamA
        # val = self.val @ streamV
        return val + adv - adv.mean()
