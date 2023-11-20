import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from agents.BaseAgent import BaseAgent
from utils import ExperienceReplayMemory, PrioritizedReplayMemory, Config

class DQN(BaseAgent):
    def __init__(self, 
                 env: dict, 
                 config: Config, 
                 log_dir='./logs',
                 static_policy=False):
        super().__init__(config=config, env=env, log_dir=log_dir, static_policy=static_policy)

        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.target_model.eval()

        # move model to device
        self.model = self.model.to(self.device)
        self.target_model.to(self.device)

        if static_policy:
            self.model.eval()
        else:
            self.model.train()

    def save(self):
        os.makedirs(self.log_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.log_dir, 'model.dump'))
        torch.save(self.optimizer.state_dict(), os.path.join(self.log_dir, 'optim.dump'))
    

    def load(self):
        fname_model = os.path.join(self.log_dir, "model.dump")
        fname_optim = os.path.join(self.log_dir, "optim.dump")

        if os.path.isfile(fname_model):
            self.model.load_state_dict(torch.load(fname_model))
            self.target_model.load_state_dict(self.model.state_dict())
        else:
            assert False

        if os.path.isfile(fname_optim):
            self.optimizer.load_state_dict(torch.load(fname_optim))
        else:
            assert False


    def declare_networks(self):
        # overload function
        self.model: nn.Module = None
        self.target_model: nn.Module = None
        raise NotImplementedError # override this function


    def declare_memory(self):
        dims = (self.num_feats, 1, 1, self.num_feats, 1)
        self.memory = ExperienceReplayMemory(self.experience_replay_size, dims) \
                        if not self.priority_replay else \
                        PrioritizedReplayMemory(self.experience_replay_size, dims, self.priority_alpha, self.priority_beta_start, self.priority_beta_frames, self.device)

    def append_to_replay(self, s, a, r, s_, done):
        self.memory.push((s, a, r, s_, done))

    def prep_minibatch(self):
        '''
        Returns:
            states: expected shape (B, S)
            actions: expected shape (B, D)
            rewards: expected shape (B, 1)
            next_states: expected shape (B, S)
            dones: expected shape (B, 1)
            indices: a list of index
            weights: expected shape (B,)
        '''
        # random transition batch is taken from replay memory
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)

        states = torch.tensor(states, device=self.device, dtype=torch.float)
        actions = torch.tensor(actions, device=self.device, dtype=torch.int64)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float)
        next_states = torch.tensor(next_states, device=self.device, dtype=torch.float)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float)

        return states, actions, rewards, next_states, dones, indices, weights


    def compute_loss(self, batch_vars):
        states, actions, rewards, next_states, dones, indices, weights = batch_vars
        self.model.train()
        q_values = self.model(states).gather(1, actions)
        with torch.no_grad():
            max_next_action = self.get_max_next_state_action(next_states)
            target_q_values = self.target_model(next_states).gather(1, max_next_action)

        if self.priority_replay:
            diff = (q_values - (rewards + self.gamma * target_q_values * (1 - dones)))
            self.memory.update_priorities(indices, diff.detach().squeeze().abs().cpu().numpy().tolist())
            loss = ((0.5 * diff.pow(2))* weights).mean()
        else:
            loss = F.mse_loss(q_values, rewards + self.gamma * target_q_values * (1 - dones))
        return loss


    def update(self, t):
        if self.static_policy:
            return None

        batch_vars = self.prep_minibatch()

        loss = self.compute_loss(batch_vars)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        if self.is_gradient_clip:
            for param in self.model.parameters():
                param.grad.data.clamp_(-1, 1) # gradient clipping, let gradient be in interval (-1, 1)
        self.optimizer.step()

        # update the target network
        if t % self.target_net_update_freq == 0:
            self.update_target_model(self.target_model, self.model)
        return {'td_error': loss.item()}


    def get_action(self, s, eps=0):
        '''
        for interacting with environment
        '''
        with torch.no_grad():
            if np.random.random() >= eps or self.static_policy:
                X = torch.tensor(np.array([s]), device=self.device, dtype=torch.float)
                a = self.model(X).max(1)[1].view(1, 1)
                return a.item()

        return np.random.randint(0, self.num_actions)

    def get_max_next_state_action(self, next_states):
        return self.model(next_states).max(dim=1)[1].view(-1, 1)


class WDQN(DQN):
    def __init__(self, 
                 env: dict, 
                 config: Config, 
                 log_dir='./logs',
                 static_policy=False):
        super().__init__(static_policy, env, config, log_dir)

    def compute_loss(self, batch_vars):
        states, actions, rewards, next_states, dones, indices, weights = batch_vars
        self.model.train()
        q_values = self.model(states).gather(1, actions)
        next_q_values = self.model(next_states)
        max_next_actions = torch.argmax(next_q_values, dim=1, keepdim=True)
        with torch.no_grad():
            target_next_q_values = self.target_model(next_states)
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
            loss = ((0.5 * diff.pow(2))* weights).mean()
        else:
            loss = F.mse_loss(q_values, rewards + self.gamma * target_q_values * (1 - dones))
        return loss
