import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from agents.BaseAgent import BaseAgent
from utils.ReplayMemory import ExperienceReplayMemory, PrioritizedReplayMemory

class Model(BaseAgent):
    def __init__(self, static_policy=False, env=None, config=None, log_dir='/log'):
        super(Model, self).__init__(config=config, env=env, log_dir=log_dir)
        self.device = config.device

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
        self.num_feats = env.num_feats
        self.num_actions = env.num_actions
        self.env = env

        # loss
        self.reg_lambda = 5
        self.reward_threshold = 20

        self.update_count = 0

        self.declare_memory()

        self.static_policy = static_policy


    def declare_networks(self, model, target_model, static_policy):
        self.model = model
        self.target_model = target_model
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        if static_policy:
            self.model.eval()
            self.target_model.eval()
        else:
            self.model.train()
            self.target_model.train()

    def move_model_to_device(self):
        # move to correct device
        self.model = self.model.to(self.device)
        self.target_model.to(self.device)

    def declare_memory(self):
        self.memory = ExperienceReplayMemory(self.experience_replay_size) if not self.priority_replay else PrioritizedReplayMemory(self.experience_replay_size, self.priority_alpha, self.priority_beta_start, self.priority_beta_frames)

    def append_to_replay(self, s, a, r, s_):
        self.memory.push((s, a, r, s_))

    def prep_minibatch(self):
        # random transition batch is taken from replay memory
        transitions, indices, weights = self.memory.sample(self.batch_size)
        
        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)

        batch_state = torch.tensor(batch_state, device=self.device, dtype=torch.float).view(-1, self.num_feats)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.int32).squeeze().view(-1, 1) # view = reshape in numpy
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).squeeze().view(-1, 1) # squeeze : delete all dimension with value = 1
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device, dtype=torch.uint8)
        try: # sometimes all next states are false
            non_final_next_states = torch.tensor([s for s in batch_next_state if s is not None], device=self.device, dtype=torch.float).view(-1, self.num_feats)
            empty_next_state_values = False
        except:
            non_final_next_states = None
            empty_next_state_values = True
            # why try have error?
            assert(True)

        return batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights

    def compute_loss(self, batch_vars): # faster
        '''
            loss function = E[Q_double-target - Q_estimate]^2 + lambda * max(|Q_estimate| - Q_threshold, 0)
            Q_double-target = reward + gamma * Q_double-target(next_state, argmax_a(Q(next_state, a)))
            Q_threshold = 20
            when die reward -15 else +15
        '''
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        # estimate
        # gather : take element of batch_action as a index of each batch_state to get the q_estimate values 
        # gather(dim, index)
        current_q_values = self.model(batch_state).gather(1, batch_action) 
        
        # target
        with torch.no_grad():
            # unnecessary to compute gradient and do back propogation
            max_next_q_values = torch.zeros(self.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)
            if not empty_next_state_values:
                max_next_action = self.get_max_next_state_action(non_final_next_states)
                max_next_q_values[non_final_mask] = self.target_model(non_final_next_states).gather(1, max_next_action)
            expected_q_values = batch_reward + ((self.gamma ** self.nsteps) * max_next_q_values)

        td_error = (expected_q_values - current_q_values).pow(2)
        if self.priority_replay:
            self.memory.update_priorities(indices, td_error.detach().squeeze().abs().cpu().numpy().tolist()) # ?
            loss = (td_error * weights).mean() + self.reg_lambda * min(current_q_values.abs().max() - self.reward_threshold, 0)
        else:
            loss = td_error.mean() + self.reg_lambda * min(current_q_values.abs().max() - self.reward_threshold, 0)

        return loss

    def update(self):
        if self.static_policy:
            return None

        batch_vars = self.prep_minibatch()

        loss = self.compute_loss(batch_vars)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        '''
        for param in self.model.parameters():
            # why clamp_ ?
            param.grad.data.clamp_(-1, 1) # clamp_ : let gradient be in interval (-1, 1)
        '''
        self.optimizer.step()

        self.update_target_model()
        # self.save_td(loss.item(), frame)
        # self.save_sigma_param_magnitudes(frame)

    def get_action(self, s, eps=0):
        with torch.no_grad():
            if np.random.random() >= eps or self.static_policy:
                X = torch.tensor([s], device=self.device, dtype=torch.float)
                a = self.model(X).max(1)[1].view(1, 1)
                return a.item()
            else:
                return np.random.randint(0, self.num_actions)

    def update_target_model(self):
        self.update_count = (self.update_count + 1) % self.target_net_update_freq
        if self.update_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def get_max_next_state_action(self, next_states):
        return self.model(next_states).max(dim=1)[1].view(-1, 1)