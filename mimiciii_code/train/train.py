import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os

#################################################################
# Network
#################################################################
class DuelingDQN(nn.Module):
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
        self.adv = nn.Linear(self.hidden2_size / 2, self.num_actions)
        self.val = nn.Linear(self.hidden2_size / 2, 1)
    

    def foward(self, x):
        y = self.l1(x)
        y = self.l2(y)
        # advantage and value streams
        streamA, streamV = torch.split(y, self.hidden2_size / 2)
        adv = self.adv(streamA)
        val = self.val(streamV)
        return val + adv - adv.mean()
