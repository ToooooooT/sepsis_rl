import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm

from utils import Config
from agents import BaseAgent
from ope.base_estimator import BaseEstimator

class FQEDataset(Dataset):
    def __init__(self,
                 states: np.ndarray,
                 actions: np.ndarray,
                 rewards: np.ndarray,
                 next_states: np.ndarray,
                 dones: np.ndarray,
                 Q: nn.Module,
                 eval_policy: BaseAgent,
                 device,
                 gamma) -> None:
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
            state: torch.tensor; expected shape (S,)
            action: torch.tensor; expected shape (1,)
            target: torch.tensor; expected shape (1,)
        '''
        state = torch.tensor(self.states[index], dtype=torch.float, device=self.device)
        action = torch.tensor(self.actions[index], dtype=torch.int64, device=self.device)
        reward = torch.tensor(self.rewards[index:index+1], dtype=torch.float, device=self.device)
        next_state = torch.tensor(self.next_states[index:index+1], dtype=torch.float, device=self.device)
        done = torch.tensor(self.dones[index:index+1], dtype=torch.float, device=self.device)
        with torch.no_grad():
            policy_action = self.eval_policy.get_action_probs(next_state)[0]
            target = reward + self.gamma * self.Q(next_state).gather(1, policy_action) * (1 - done)
        return state, action, target.detach().view(-1)



class FQE(BaseEstimator):
    def __init__(self, 
                 agent: BaseAgent, 
                 data_dict: dict, 
                 config: Config, 
                 args,
                 Q: nn.Module,
                 target_Q: nn.Module,
                 lr=1e-4,
                 batch_size=256,
                 episode=2e3) -> None:
        # ref: Batch Policy Learning under Constraints
        super().__init__(agent, data_dict, config, args)

        self.Q = Q.to(config.DEVICE)
        self.target_Q = target_Q.to(config.DEVICE)
        self.target_Q.eval()
        self.lr = lr
        self.batch_size = batch_size
        self.episode = int(episode)
        self.optimizer = optim.Adam(self.Q.parameters(), lr=self.lr)
        self.is_gradient_clip = config.IS_GRADIENT_CLIP
        self.num_worker = args.num_worker
        self.records = pd.DataFrame({'episode': [], 'epoch_loss': []})

    def estimate(self, **kwargs):
        '''
        Returns:
            average policy return
        '''
        self.reset()
        self.train()

        done_indexs = np.where(self.dones == 1)[0]
        start_indexs = [0] + (done_indexs + 1).tolist()[:-1]
        states = torch.tensor(self.states[start_indexs], dtype=torch.float, device=self.device)
        with torch.no_grad():
            actions = self.agent.get_action_probs(states)[0]
            expected_return = self.Q(states).gather(1, actions)
        return expected_return.mean().item(), expected_return.reshape(1, -1).cpu().numpy()


    def train(self):
        self.Q.train()
        for i in tqdm(range(self.episode)):
            fqe_dataset = FQEDataset(self.states,
                                     self.actions,
                                     self.rewards,
                                     self.next_states,
                                     self.dones,
                                     self.target_Q,
                                     self.agent,
                                     self.device,
                                     self.gamma)
            dataloader = DataLoader(fqe_dataset,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    drop_last=False,
                                    pin_memory=True,
                                    num_workers=1)
            epoch_loss = 0
            for state, action, label in dataloader:
                pred = self.Q(state).gather(1, action)
                loss = F.mse_loss(pred, label)
                self.optimizer.zero_grad()
                loss.backward()
                if self.is_gradient_clip:
                    for param in self.Q.parameters():
                        param.grad.data.clamp_(-1, 1) # gradient clipping, let gradient be in interval (-1, 1)
                self.optimizer.step()
                epoch_loss += loss.item()
            self.records.loc[self.records.shape[0]] = {'episode': i, 'epoch_loss': epoch_loss}
            self.update_target_model(self.target_Q, self.Q)

    def reset(self):
        self.Q.initialize()
        self.target_Q.load_state_dict(self.Q.state_dict())
        self.optimizer = optim.Adam(self.Q.parameters(), lr=self.lr)

    def update_target_model(self, target: nn.Module, source: nn.Module):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


    def records2csv(self):
        path = os.path.join(self.agent.log_dir, 'fqe_loss.csv')
        self.records.to_csv(path, index=False)
