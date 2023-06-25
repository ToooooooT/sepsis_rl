import numpy as np

import torch
import torch.optim as optim

from agents.BaseAgent import BaseAgent
from utils.ReplayMemory import ExperienceReplayMemory, PrioritizedReplayMemory

class Model(BaseAgent):
    def __init__(self, static_policy=False, env=None, config=None, log_dir='./log', agent_dir='./saved_agents'):
        super(Model, self).__init__(config=config, env=env, log_dir=log_dir, agent_dir=agent_dir)

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
        self.model = None
        self.target_model = None
        raise NotImplementedError


    def declare_memory(self):
        self.memory = ExperienceReplayMemory(self.experience_replay_size) if not self.priority_replay else PrioritizedReplayMemory(self.experience_replay_size, self.priority_alpha, self.priority_beta_start, self.priority_beta_frames)

    def append_to_replay(self, s, a, r, s_):
        self.memory.push((s, a, r, s_))

    def prep_minibatch(self):
        '''
        Returns:
            batch_state: expected shape (B, S)
            batch_action: expected shape (B, D)
            batch_reward: expected shape (B, 1)
            non_final_next_state: expected shape (., D)
            non_final_mask: true if it is not final state; expected shape (B)
            empty_next_state_values: expected shape
            indices: a list of index
            weights: expected shape (B,)
        '''
        # random transition batch is taken from replay memory
        transitions, indices, weights = self.memory.sample(self.batch_size)
        
        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)

        batch_state = np.array(batch_state)
        batch_action = np.array(batch_action)
        batch_reward = np.array(batch_reward)

        batch_state = torch.tensor(batch_state, device=self.device, dtype=torch.float).view(-1, self.num_feats)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.int64).squeeze().view(-1, 1) # view = reshape in numpy
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).squeeze().view(-1, 1) # squeeze : delete all dimension with value = 1
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device, dtype=torch.bool)
        try: # sometimes all next states are false
            non_final_next_states = torch.tensor(np.array([s for s in batch_next_state if s is not None]), device=self.device, dtype=torch.float).view(-1, self.num_feats)
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
        q_values = self.model(batch_state)
        current_q_values = q_values.gather(1, batch_action) 
        
        # compute target value
        with torch.no_grad():
            # unnecessary to compute gradient and do back propogation
            max_next_q_values = torch.zeros(self.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)
            if not empty_next_state_values:
                max_next_action = self.get_max_next_state_action(non_final_next_states)
                max_next_q_values[non_final_mask] = self.target_model(non_final_next_states).gather(1, max_next_action)

            # empirical hack to make the Q values never exceed the threshold - helps learning
            if self.reg_lambda > 0:
                max_next_q_values[max_next_q_values > self.reward_threshold] = self.reward_threshold
                max_next_q_values[max_next_q_values < -self.reward_threshold] = -self.reward_threshold

            expected_q_values = batch_reward + ((self.gamma ** self.nsteps) * max_next_q_values)

        td_error = (expected_q_values - current_q_values).pow(2)
        if self.priority_replay:
            self.memory.update_priorities(indices, (td_error / 10).detach().squeeze().abs().cpu().numpy().tolist()) # ?
            loss = 0.5 * (td_error * weights).mean() + self.reg_lambda * max(current_q_values.abs().max() - self.reward_threshold, 0)
            # loss = 0.5 * (td_error * weights).mean()
        else:
            loss = 0.5 * td_error.mean() + self.reg_lambda * max(current_q_values.abs().max() - self.reward_threshold, 0)
            # loss = 0.5 * td_error.mean()
        return loss

    def update(self, t):
        if self.static_policy:
            return None

        batch_vars = self.prep_minibatch()

        loss = self.compute_loss(batch_vars)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1) # clamp_ : let gradient be in interval (-1, 1)
        self.optimizer.step()

        # update the target network
        if t % self.target_net_update_freq == 0:
            self.update_target_model()
        return loss.item()

    def get_action(self, s, eps=0):
        with torch.no_grad():
            if np.random.random() >= eps or self.static_policy:
                X = torch.tensor([s], device=self.device, dtype=torch.float)
                a = self.model(X).max(1)[1].view(1, 1)
                return a.item()
            else:
                return np.random.randint(0, self.num_actions)

    def update_target_model(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def get_max_next_state_action(self, next_states):
        return self.model(next_states).max(dim=1)[1].view(-1, 1)