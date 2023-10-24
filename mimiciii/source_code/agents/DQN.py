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

        # step
        self.nsteps = 1

        # algorithm control
        self.priority_replay = config.USE_PRIORITY_REPLAY

        # misc agent variables
        self.gamma = config.GAMMA
        self.lr = config.LR

        # memory
        self.target_net_update_freq = config.TARGET_NET_UPDATE_FREQ
        self.experience_replay_size = config.EXP_REPLAY_SIZE
        self.batch_size = config.BATCH_SIZE
        self.priority_alpha = config.PRIORITY_ALPHA
        self.priority_beta_start = config.PRIORITY_BETA_START
        self.priority_beta_frames = config.PRIORITY_BETA_FRAMES

        # environment
        self.num_feats = env['num_feats']
        self.num_actions = env['num_actions']

        # loss
        self.reg_lambda = config.REG_LAMBDA
        self.reward_threshold = 20

        self.update_count = 0

        self.declare_memory()

        # network
        self.static_policy = static_policy

        self.declare_networks()
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # update target network parameter
        self.tau = 0.001

        if static_policy:
            self.model.eval()
            self.target_model.eval()
        else:
            self.model.train()
            self.target_model.train()

        self.model = self.model.to(self.device)
        self.target_model.to(self.device)


    def declare_networks(self):
        # overload function
        self.model: nn.Module = None
        self.target_model: nn.Module = None
        raise NotImplementedError


    def declare_memory(self):
        self.memory = ExperienceReplayMemory(self.experience_replay_size) if not self.priority_replay else PrioritizedReplayMemory(self.experience_replay_size, self.priority_alpha, self.priority_beta_start, self.priority_beta_frames)

    def append_to_replay(self, s, a, r, s_, done):
        self.memory.push((s, a, r, s_, done))

    def prep_minibatch(self):
        '''
        Returns:
            batch_state: expected shape (B, S)
            batch_action: expected shape (B, D)
            batch_reward: expected shape (B, 1)
            batch_next_state: expected shape (B, S)
            batch_done: expected shape (B, 1)
            indices: a list of index
            weights: expected shape (B,)
        '''
        # random transition batch is taken from replay memory
        transitions, indices, weights = self.memory.sample(self.batch_size)
        
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

        batch_state = torch.tensor(np.array(batch_state), device=self.device, dtype=torch.float)
        batch_action = torch.tensor(np.array(batch_action), device=self.device, dtype=torch.int64).view(-1, 1)
        batch_reward = torch.tensor(np.array(batch_reward), device=self.device, dtype=torch.float).view(-1, 1)
        batch_next_state = torch.tensor(np.array(batch_next_state), device=self.device, dtype=torch.float)
        batch_done = torch.tensor(np.array(batch_done), device=self.device, dtype=torch.float).view(-1, 1)
        
        return batch_state, batch_action, batch_reward, batch_next_state, batch_done, indices, weights


    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, batch_next_state, batch_done, indices, weights = batch_vars
        self.model.train()
        q_values = self.model(batch_state).gather(1, batch_action)
        with torch.no_grad():
            max_next_action = self.get_max_next_state_action(batch_next_state)
            target_q_values = self.target_model(batch_next_state).gather(1, max_next_action)

        if self.priority_replay:
            td_error = (q_values - target_q_values).pow(2)
            self.memory.update_priorities(indices, (td_error / 10).detach().squeeze().abs().cpu().numpy().tolist()) # ?
            loss = 0.5 * (td_error * weights).mean()
        else:
            loss = F.mse_loss(q_values, batch_reward + self.gamma * target_q_values * (1 - batch_done))
        return loss


    def update(self, t):
        if self.static_policy:
            return None

        batch_vars = self.prep_minibatch()

        loss = self.compute_loss(batch_vars)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.model.parameters():
        #     param.grad.data.clamp_(-1, 1) # clamp_ : let gradient be in interval (-1, 1)
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
        batch_state, batch_action, batch_reward, batch_next_state, batch_done, indices, weights = batch_vars
        self.model.train()
        q_values = self.model(batch_state).gather(1, batch_action)
        next_q_values = self.model(batch_next_state)
        max_next_actions = torch.argmax(next_q_values, dim=1)
        with torch.no_grad():
            target_next_q_values = self.target_model(batch_next_state)
        max_target_next_actions = torch.argmax(target_next_q_values, dim=1)
        target_next_q_values_softmax = F.softmax(target_next_q_values)
        sigma = target_next_q_values_softmax.gather(1, max_next_actions)
        phi = target_next_q_values_softmax.gather(1, max_target_next_actions)
        p = phi / (phi + sigma)
        target_q_values = p * target_next_q_values.max(dim=1) + (1 - p) * target_next_q_values.gather(1, max_next_actions)

        if self.priority_replay:
            td_error = (q_values - target_q_values).pow(2)
            self.memory.update_priorities(indices, (td_error / 10).detach().squeeze().abs().cpu().numpy().tolist()) # ?
            loss = 0.5 * (td_error * weights).mean()
        else:
            loss = F.mse_loss(q_values, batch_reward + self.gamma * target_q_values * (1 - batch_done))
        return loss
