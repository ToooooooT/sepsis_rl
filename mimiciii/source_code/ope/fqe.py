from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('../')

from agents import BaseAgent


class FQEDataset(Dataset):
    def __init__(self,
                 states: np.ndarray,
                 actions: np.ndarray,
                 next_states: np.ndarray,
                 rewards: np.ndarray,
                 dones: np.ndarray,
                 Q: nn.Module,
                 eval_policy: BaseAgent,
                 device,
                 gamma=5e-3) -> None:
        '''
        Args:
            states: expected shape (B, S)
            actions: expected shape (B, D)
            next_states: expected shape (B, S)
            reward: expected shape (B, 1)
            dones: expected shape (B, 1)
            Q: last step Q function
            eval_policy: 
        '''
        super().__init__()
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.dones = dones
        self.Q = Q.to(device)
        self.eval_policy = eval_policy
        self.device = device
        self.gamma = gamma

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, index):
        '''
        Returns: 
            state: torch.tensor; expected shape (1, S)
            action: torch.tensor; expected shape (1, 1)
            target: torch.tensor; expected shape (1, 1)
        '''
        state = torch.tensor(self.states[index], dtype=torch.float, device=self.device)
        action = torch.tensor(self.actions[index], dtype=torch.float, device=self.device)
        reward = torch.tensor(self.rewards[index], dtype=torch.float, device=self.device)
        next_state = torch.tensor(self.next_states[index], dtype=torch.float, device=self.device)
        done = torch.tensor(self.dones[index], dtype=torch.float, device=self.device)
        # target = reward + self.gamma * self.Q(next_state).max(dim=1)[0].view(-1, 1) * (1 - done)
        # return state, action, target



class FQE():
    def __init__(self,
                 states: np.ndarray,
                 actions: np.ndarray,
                 next_states: np.ndarray,
                 rewards: np.ndarray,
                 dones: np.ndarray,
                 eval_policy: BaseAgent,
                 Q: nn.Module, 
                 device,
                 gamma=5e-3,
                 episode=1e3) -> None:
        self.states = states
        self.actions = actions
        self.next_states = next_states
        self.rewards = rewards
        self.dones = dones
        self.eval_policy = eval_policy
        # self.Q.
        pass

