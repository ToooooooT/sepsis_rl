import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Tuple

from agents.BaseAgent import BaseAgent
from utils import Config
from network import WDQN_DuelingMLP

class DQN(BaseAgent):
    def __init__(self, 
                 env: dict, 
                 config: Config, 
                 log_dir='./logs',
                 static_policy=False):
        super().__init__(config=config, env=env, log_dir=log_dir, static_policy=static_policy)

        self.target_q.load_state_dict(self.q.state_dict())
        self.optimizer = optim.Adam(self.q.parameters(), lr=self.q_lr)
        self.target_q.eval()

        # move model to device
        self.q = self.q.to(self.device)
        self.target_q.to(self.device)

        if static_policy:
            self.q.eval()
        else:
            self.q.train()

    def save_checkpoint(self, epoch: int, name: str='checkpoint.pth'):
        checkpoint = {
            'epoch': epoch,
            'model': self.q.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(self.log_dir, name))

    def load_checkpoint(self, name='checkpoint.pth') -> int:
        path = os.path.join(self.log_dir, name)
        if os.path.exists(path):
            checkpoint = torch.load(path)
        else:
            raise FileExistsError
        self.q.load_state_dict(checkpoint['model'])
        self.target_q.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint['epoch']

    def save(self, name: str='model.pth'):
        os.makedirs(self.log_dir, exist_ok=True)
        torch.save(self.q.state_dict(), os.path.join(self.log_dir, name))
    

    def load(self, name: str='model.pth'):
        fname_model = os.path.join(self.log_dir, name)

        if os.path.isfile(fname_model):
            self.q.load_state_dict(torch.load(fname_model))
            self.target_q.load_state_dict(self.q.state_dict())
        else:
            assert False

    def compute_loss(self, batch_vars: Tuple) -> torch.Tensor:
        states, actions, rewards, next_states, dones, indices, weights = batch_vars

        if self.use_state_augmentation:
            states, next_states = self.augmentation(states, next_states, rewards, dones)
            actions = actions.unsqueeze(1).repeat(1, 2, 1)

        q_values = self.q(states).gather(-1, actions)
        with torch.no_grad():
            max_next_action = self.get_max_next_state_action(next_states)
            target_q_values = self.target_q(next_states).gather(-1, max_next_action)

        if self.use_state_augmentation:
            q_values = q_values.mean(dim=1)
            target_q_values = target_q_values.mean(dim=1)

        if self.priority_replay:
            diff = (q_values - (rewards + self.gamma * target_q_values * (1 - dones)))
            self.memory.update_priorities(indices, diff.detach().squeeze().abs().cpu().numpy().tolist())
            loss = ((0.5 * diff.pow(2))* weights).mean()
        else:
            loss = F.mse_loss(q_values, rewards + self.gamma * target_q_values * (1 - dones))
        return loss


    def update(self, t: int) -> Dict:
        if self.static_policy:
            return None

        self.q.train()
        batch_vars = self.prep_minibatch()

        loss = self.compute_loss(batch_vars)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        if self.is_gradient_clip:
            for param in self.q.parameters():
                param.grad.data.clamp_(-1, 1) # gradient clipping, let gradient be in interval (-1, 1)
        self.optimizer.step()

        # update the target network
        if t % self.target_net_update_freq == 0:
            self.update_target_model(self.target_q, self.q)
        return {'td_error': loss.item()}


    def get_action(self, s, eps=0) -> int:
        '''
        for interacting with environment
        '''
        with torch.no_grad():
            if np.random.random() >= eps or self.static_policy:
                X = torch.tensor(np.array([s]), device=self.device, dtype=torch.float)
                a = self.q(X).max(1)[1].view(1, 1)
                return a.item()

        return np.random.randint(0, self.num_actions)

    def get_action_probs(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _, actions = self.q(states).max(dim=1)
        actions = actions.view(-1, 1) # (B, 1)
        action_probs = torch.full((actions.shape[0], 25), 0.01, device=self.device)
        action_probs = action_probs.scatter_(1, actions, 0.99)
        return actions, None, None, action_probs

    def get_max_next_state_action(self, next_states: torch.Tensor) -> torch.Tensor:
        return self.q(next_states).max(dim=-1, keepdim=True)[1]

    def adversarial_state_training(self, 
                                   states: np.ndarray, 
                                   next_states: np.ndarray, 
                                   rewards: np.ndarray,
                                   dones: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''
        for data augmentation
        '''
        # TODO: check this function, currently only augment on states
        states = torch.tensor(states, device=self.device, dtype=torch.float, requires_grad=True)
        next_states = torch.tensor(next_states, device=self.device, dtype=torch.float)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float)

        q_values = self.q(states).max(1, keepdim=True)[0]
        next_q_values = self.q(next_states).max(1, keepdim=True)[0]
        loss = (rewards + next_q_values * (1 - dones) - q_values).mean()
        states.grad.zero_()
        loss.backward()
        with torch.no_grad():
            states = states + states.grad * self.adversarial_step
        return states.detach().cpu().numpy(), next_states.cpu().numpy()

        
class WDQN(DQN):
    def __init__(self, 
                 env: dict, 
                 config: Config, 
                 log_dir='./logs',
                 static_policy=False):
        super().__init__(env, config, log_dir, static_policy)

    def declare_networks(self):
        self.q = WDQN_DuelingMLP(self.num_feats, self.num_actions).to(self.device)
        self.target_q = WDQN_DuelingMLP(self.num_feats, self.num_actions).to(self.device)

    def compute_loss(self, batch_vars: Tuple) -> torch.Tensor:
        # TODO: state augmentaton
        states, actions, rewards, next_states, dones, indices, weights = batch_vars
        q_values = self.q(states).gather(1, actions)
        next_q_values = self.q(next_states)
        max_next_actions = torch.argmax(next_q_values, dim=1, keepdim=True)
        with torch.no_grad():
            target_next_q_values = self.target_q(next_states)
        max_target_next_actions = torch.argmax(target_next_q_values, dim=1, keepdim=True)
        target_next_q_values_softmax = F.softmax(target_next_q_values, dim=1)
        sigma = target_next_q_values_softmax.gather(1, max_next_actions)
        phi = target_next_q_values_softmax.gather(1, max_target_next_actions)
        p = phi / (phi + sigma)
        target_q_values = p * target_next_q_values.max(dim=1)[0].view(-1, 1) + \
                            (1 - p) * target_next_q_values.gather(1, max_next_actions)

        if self.priority_replay:
            diff = (q_values - (rewards + self.gamma * target_q_values * (1 - dones)))
            self.memory.update_priorities(indices, diff.detach().squeeze().abs().cpu().numpy().tolist())
            loss = (F.smooth_l1_loss(q_values, 
                                    rewards + self.gamma * target_q_values * (1 - dones),
                                    reduction='none') * weights).mean()
        else:
            loss = F.smooth_l1_loss(q_values, 
                                    rewards + self.gamma * target_q_values * (1 - dones))
        return loss
