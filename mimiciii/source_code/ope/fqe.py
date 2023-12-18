import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
from typing import Tuple

from utils import Config
from agents import BaseAgent
from ope.base_estimator import BaseEstimator

class FQEDataset(Dataset):
    def __init__(self, train_dict: dict) -> None:
        '''
        Args:
            states: expected shape (B, S)
            actions: expected shape (B, D)
            next_states: expected shape (B, S)
            reward: expected shape (B, 1)
            dones: expected shape (B, 1)
        '''
        super().__init__()
        self.states = train_dict['s']
        self.actions = train_dict['a']
        self.rewards = train_dict['r']
        self.next_states = train_dict['s_']
        self.dones = train_dict['done']

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, index):
        '''
        Returns: 
            state: torch.tensor; expected shape (S,)
            action: torch.tensor; expected shape (1,)
            reward: torch.tensor; expected shape (1,)
            next_state: torch.tensor; expected shape (S,)
            done: torch.tensor; expected shape (1,)
        '''
        state = torch.tensor(self.states[index], dtype=torch.float)
        action = torch.tensor(self.actions[index], dtype=torch.int64)
        reward = torch.tensor(self.rewards[index], dtype=torch.float)
        next_state = torch.tensor(self.next_states[index], dtype=torch.float)
        done = torch.tensor(self.dones[index], dtype=torch.float)
        return state, action, reward, next_state, done



class FQE(BaseEstimator):
    def __init__(self, 
                 agent: BaseAgent, 
                 train_dict: dict, 
                 test_dict: dict, 
                 config: Config, 
                 args,
                 Q: nn.Module,
                 target_Q: nn.Module) -> None:
        # ref: Batch Policy Learning under Constraints
        super().__init__(agent, test_dict, config, args)
        self.train_dict = train_dict
        self.Q = Q.to(config.DEVICE)
        self.target_Q = target_Q.to(config.DEVICE)
        self.target_Q.eval()
        self.lr = args.fqe_lr
        self.batch_size = args.fqe_batch_size
        self.episode = int(args.fqe_episode)
        self.optimizer = optim.Adam(self.Q.parameters(), lr=self.lr)
        self.is_gradient_clip = config.IS_GRADIENT_CLIP
        self.num_worker = args.num_worker
        self.records = pd.DataFrame({'episode': [], 'epoch_loss': []})

        done_indexs = np.where(self.dones == 1)[0]
        start_indexs = [0] + (done_indexs + 1).tolist()[:-1]
        self.initial_states = torch.tensor(self.states[start_indexs], dtype=torch.float)

        # train dataset
        self.dataloader = DataLoader(FQEDataset(self.train_dict),
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    drop_last=False,
                                    pin_memory=True,
                                    num_workers=self.num_worker)

    def estimate(self, **kwargs):
        '''
        Returns:
            average policy return
        '''
        self.agent = kwargs['agent']
        self.reset()
        self.train()

        return self.predict()


    def predict(self) -> Tuple[float, np.ndarray]:
        initial_states = self.initial_states.to(self.device)
        with torch.no_grad():
            actions = self.agent.get_action_probs(initial_states)[0]
            expected_return = self.Q(initial_states).gather(1, actions)
        return expected_return.mean().item(), expected_return.reshape(1, -1).cpu().numpy()


    def train(self):
        self.Q.train()
        M = 5
        estimate_values = []
        for i in tqdm(range(self.episode)):
            epoch_loss = 0
            for state, action, reward, next_state, done in self.dataloader:
                state = state.to(self.device)
                action = action.to(self.device)
                reward = reward.to(self.device)
                next_state = next_state.to(self.device)
                done = done.to(self.device)
                with torch.no_grad():
                    policy_action = self.agent.get_action_probs(next_state)[0]
                    target = reward + self.gamma * self.target_Q(next_state).gather(1, policy_action) * (1 - done)
                pred = self.Q(state).gather(1, action)
                loss = F.mse_loss(pred, target.detach())
                self.optimizer.zero_grad()
                loss.backward()
                if self.is_gradient_clip:
                    for param in self.Q.parameters():
                        param.grad.data.clamp_(-1, 1) # gradient clipping, let gradient be in interval (-1, 1)
                self.optimizer.step()
                epoch_loss += loss.item()

            estimate_values.append(self.predict()[0])
            if i > M and                                                                        \
                np.abs(np.mean(estimate_values[-M:]) - np.mean(estimate_values[-(M + 1):-1]))   \
                    < 1e-4 * np.abs(np.mean(estimate_values[-(M+1):-1])):
                break

            self.records.loc[self.records.shape[0]] = {'episode': i, 
                                                       'epoch_loss': epoch_loss,
                                                       'estimate_value': estimate_values[-1]}

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
