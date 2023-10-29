import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from agents.BaseAgent import BaseAgent
from utils.ReplayMemory import ExperienceReplayMemory, PrioritizedReplayMemory

class DQN(BaseAgent):
    def __init__(self, static_policy=False, env=None, config=None, log_dir='./logs'):
        super().__init__(config=config, env=env, log_dir=log_dir)
        self.device = config.device

        # algorithm control
        self.priority_replay = config.USE_PRIORITY_REPLAY

        # misc agent variables
        self.gamma = config.GAMMA
        self.lr = config.LR

        self.is_gradient_clip = config.IS_GRADIENT_CLIP

        # memory
        self.target_net_update_freq = config.TARGET_NET_UPDATE_FREQ
        self.experience_replay_size = config.EXP_REPLAY_SIZE
        self.batch_size = config.BATCH_SIZE
        self.priority_alpha = config.PRIORITY_ALPHA
        self.priority_beta_start = config.PRIORITY_BETA_START
        self.priority_beta_frames = config.PRIORITY_BETA_FRAMES

        # update target network parameter
        self.tau = config.TAU

        # environment
        self.num_feats = env['num_feats']
        self.num_actions = env['num_actions']

        self.declare_memory()

        # network
        self.static_policy = static_policy

        self.declare_networks()
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


    def declare_networks(self):
        # overload function
        self.model: nn.Module = None
        self.target_model: nn.Module = None
        raise NotImplementedError # override thid function


    def declare_memory(self):
        self.memory = ExperienceReplayMemory(self.experience_replay_size) if not self.priority_replay else PrioritizedReplayMemory(self.experience_replay_size, self.priority_alpha, self.priority_beta_start, self.priority_beta_frames)

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
        transitions, indices, weights = self.memory.sample(self.batch_size)
        
        states, actions, rewards, next_states, dones = zip(*transitions)

        states = torch.tensor(np.array(states), device=self.device, dtype=torch.float)
        actions = torch.tensor(np.array(actions), device=self.device, dtype=torch.int64).view(-1, 1)
        rewards = torch.tensor(np.array(rewards), device=self.device, dtype=torch.float).view(-1, 1)
        next_states = torch.tensor(np.array(next_states), device=self.device, dtype=torch.float)
        dones = torch.tensor(np.array(dones), device=self.device, dtype=torch.float).view(-1, 1)

        # check shape
        assert states.shape == (self.batch_size, self.num_feats)
        assert actions.shape == (self.batch_size, 1)
        assert rewards.shape == (self.batch_size, 1)
        assert next_states.shape == (self.batch_size, self.num_feats)
        assert dones.shape == (self.batch_size, 1)

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
            self.update_target_model()
        return {'td_error': loss.item()}


    def get_action(self, s, eps=0):
        with torch.no_grad():
            if np.random.random() >= eps or self.static_policy:
                X = torch.tensor(np.array([s]), device=self.device, dtype=torch.float)
                a = self.model(X).max(1)[1].view(1, 1)
                return a.item()
            else:
                return np.random.randint(0, self.num_actions)

    def update_target_model(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def get_max_next_state_action(self, next_states):
        return self.model(next_states).max(dim=1)[1].view(-1, 1)


class WDQN(DQN):
    def __init__(self, static_policy=False, env=None, config=None, log_dir='./logs'):
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
